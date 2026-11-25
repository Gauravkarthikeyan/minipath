"""The single controller.

One Python process holds the whole program. It compiles functions once and
ships them to workers, then dispatches operations that consume and produce
*futures* (buffer ids that live on the workers). Dispatch never waits for
compute: ``run`` returns as soon as the op message is on the wire. Tensors
move worker-to-worker. The controller only sees the bytes you ``fetch``.
"""

from __future__ import annotations

import itertools
import os
import random
import secrets
import signal
import sys
import threading
import time
from collections import deque
from multiprocessing import get_context
from multiprocessing.connection import Listener

import cloudpickle

from .resources import ResourceManager, Slice
from .worker import worker_main

WORKER_ENV = {
    # One XLA thread per worker, so N workers on N cores behave like N devices.
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "JAX_PLATFORMS": "cpu",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}


class WorkerLost(RuntimeError):
    """A worker died. Dispatch is refused until ``begin_recovery`` is called."""


class RemoteOpError(RuntimeError):
    pass


class Buffer:
    """A future for one tensor (or pytree) living on one worker."""

    __slots__ = ("__weakref__", "cluster", "key", "wid")

    def __init__(self, cluster: Cluster, wid: int, key: int):
        self.cluster, self.wid, self.key = cluster, wid, key

    def fetch(self, timeout: float | None = 120.0):
        return self.cluster._fetch(self.wid, self.key, timeout)

    def __del__(self):
        c = self.cluster
        if c is not None and not c._closed:
            c._garbage.append((self.wid, self.key))

    def __repr__(self) -> str:
        return f"Buffer(w{self.wid}:{self.key})"


class Sharded:
    """A future for a value sharded (or replicated) over a slice, one buffer per rank."""

    def __init__(self, slice_: Slice, buffers: list):
        self.slice = slice_
        self.buffers = buffers

    def __len__(self) -> int:
        return len(self.buffers)

    def __getitem__(self, i) -> Sharded:
        if isinstance(i, slice):
            return Sharded(self.slice[i], self.buffers[i])
        return Sharded(self.slice[i], [self.buffers[i]])

    def fetch(self, rank: int | None = None, timeout: float | None = 120.0):
        if rank is not None:
            return self.buffers[rank].fetch(timeout)
        return [b.fetch(timeout) for b in self.buffers]

    def __repr__(self) -> str:
        return f"Sharded({self.buffers})"


class Program:
    _ids = itertools.count()

    def __init__(self, fn, name: str, jit: bool, n_out: int):
        self.id = next(Program._ids)
        self.name = name
        self.jit = jit
        self.n_out = n_out
        self.payload = cloudpickle.dumps(fn)
        self.compiled_on: set = set()


class Cluster:
    """Starts workers and is the single client that drives them.

    ``gang_scheduling=True`` runs every multi-worker op through a central
    sequencer, so all members see ops in the same global order. Turn it off
    to see what happens to independent clients that share devices.
    """

    def __init__(
        self,
        n_workers: int,
        *,
        n_spares: int = 0,
        gang_scheduling: bool = True,
        shuffle_unscheduled: bool = True,
        env: dict | None = None,
        start_timeout: float = 60.0,
        verbose: bool = False,
    ):
        self.n_total = n_workers + n_spares
        self.gang_scheduling = gang_scheduling
        self.shuffle_unscheduled = shuffle_unscheduled
        self.verbose = verbose
        self._closed = False
        self._authkey = secrets.token_bytes(16)
        self._listener = Listener(family="AF_UNIX", authkey=self._authkey)
        self._ctx = get_context("spawn")
        self._sched_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._send_locks = {w: threading.Lock() for w in range(self.n_total)}
        self._ids = itertools.count(1)
        self._garbage: deque = deque()
        self._pending: dict = {}
        self._done: dict = {}
        self._epoch = 0
        self._needs_recovery = False
        self._lost: list = []
        self.failure_time: float | None = None
        self.errors: list = []
        self.spmd_logs: list = []
        self._spmd_results: dict = {}
        self._spmd_event = threading.Event()
        self.alive: set = set()

        env = {**WORKER_ENV, **(env or {})}
        self.procs = {}
        for w in range(self.n_total):
            p = self._ctx.Process(target=worker_main, args=(w, self._listener.address, self._authkey, env), daemon=True)
            p.start()
            self.procs[w] = p

        self.conns, self.pids, peer_addrs = {}, {}, {}
        started = threading.Event()

        def watchdog():  # accept() has no timeout; unblock it if workers never arrive
            if not started.wait(start_timeout):
                for p in self.procs.values():
                    p.kill()
                self._listener.close()

        threading.Thread(target=watchdog, daemon=True).start()
        while len(self.conns) < self.n_total:
            try:
                conn = self._listener.accept()
            except OSError as e:
                raise TimeoutError(f"workers failed to start within {start_timeout}s") from e
            _, wid, addr, pid = conn.recv()
            self.conns[wid], self.pids[wid], peer_addrs[wid] = conn, pid, addr
        started.set()
        self.alive = set(self.conns)
        for w, conn in self.conns.items():
            conn.send(("peers", peer_addrs))
            threading.Thread(target=self._reader, args=(w,), daemon=True).start()

        self.rm = ResourceManager(workers=list(range(n_workers)), spares=list(range(n_workers, self.n_total)))

    # ------------------------------------------------------------------ io

    def _send(self, w: int, msg) -> None:
        with self._send_locks[w]:
            self.conns[w].send(msg)

    def _reader(self, w: int) -> None:
        conn = self.conns[w]
        try:
            while True:
                self._handle(w, conn.recv())
        except (EOFError, OSError):
            self._on_lost(w)

    def _handle(self, w: int, msg) -> None:
        kind = msg[0]
        if kind in ("value", "value_error"):
            slot = self._pending.pop(msg[1], None)
            if slot is not None:
                slot[1] = (kind, msg[2])
                slot[0].set()
        elif kind == "done":
            entry = self._done.get(msg[1])
            if entry is not None:
                with self._state_lock:
                    entry[0] -= 1
                    if entry[0] == 0:
                        entry[1].set()
        elif kind == "error":
            self.errors.append((msg[1], msg[2], msg[3]))
            if self.verbose:
                print(f"[w{msg[2]}] op {msg[1]} failed:\n{msg[3]}", file=sys.stderr)
        elif kind == "log":
            print(f"[w{msg[1]}] {msg[2]}", file=sys.stderr)
        elif kind == "spmd_log":
            self.spmd_logs.append((time.perf_counter(), msg[1], msg[2], msg[3]))
        elif kind in ("spmd_done", "spmd_error"):
            self._spmd_results[w] = (kind, msg[2])
            self._spmd_event.set()

    def _on_lost(self, w: int) -> None:
        if self._closed:
            return
        with self._state_lock:
            if w not in self.alive:
                return
            self.alive.discard(w)
            self.failure_time = time.perf_counter()
            vdevs = self.rm.mark_dead(w)
            self._lost.append((w, vdevs))
            self._needs_recovery = True
            self._epoch += 1
            epoch = self._epoch
        # Tell every survivor to drop in-flight work. This is out-of-band, so
        # it also unblocks a worker that is stuck inside a collective.
        for s in list(self.alive):
            try:
                self._send(s, ("abort", epoch))
            except OSError:
                pass
        for req, slot in list(self._pending.items()):
            if slot[2] == w:
                self._pending.pop(req, None)
                slot[1] = ("lost", w)
                slot[0].set()
        for entry in list(self._done.values()):
            entry[1].set()
        self._spmd_event.set()

    def _flush_garbage(self) -> None:
        if not self._garbage:
            return
        by_worker: dict = {}
        while self._garbage:
            try:
                w, k = self._garbage.popleft()
            except IndexError:
                break
            by_worker.setdefault(w, []).append(k)
        for w, keys in by_worker.items():
            if w in self.alive:
                self._send(w, ("delete", keys))

    def _check(self) -> None:
        if self._needs_recovery:
            raise WorkerLost(f"lost workers {[w for w, _ in self._lost]}; call begin_recovery()")

    def _dispatch(self, msgs: list) -> None:
        """Send one logical op to every member of its gang.

        With gang scheduling the whole gang is sent under one lock, so every
        op has a single global position, and two gangs that share a worker
        are seen in the same order by every worker they share.
        """
        if self.gang_scheduling:
            with self._sched_lock:
                self._check()
                self._flush_garbage()
                for w, m in msgs:
                    self._send(w, m)
        else:
            self._check()
            self._flush_garbage()
            # Independent clients: each worker gets its own messages in order,
            # but nothing orders one client's gang against another's.
            per_worker: dict = {}
            for w, m in msgs:
                per_worker.setdefault(w, []).append(m)
            order = list(per_worker)
            if self.shuffle_unscheduled:
                random.shuffle(order)
            for w in order:
                for m in per_worker[w]:
                    self._send(w, m)
                time.sleep(0)

    def _fetch(self, w: int, key: int, timeout: float | None):
        if w not in self.alive:
            raise WorkerLost(f"worker {w} is dead")
        ev = threading.Event()
        req = next(self._ids)
        slot = [ev, None, w]
        self._pending[req] = slot
        self._send(w, ("fetch", req, key))
        if not ev.wait(timeout):
            self._pending.pop(req, None)
            raise TimeoutError(f"fetch of {key} from worker {w} timed out")
        kind, payload = slot[1]
        if kind == "value":
            return payload
        if kind == "lost":
            raise WorkerLost(f"worker {payload} died before {key} was ready")
        raise RemoteOpError(payload)

    def _request(self, w: int, msg_fn, timeout: float = 60.0):
        """Queue a request op in ``w``'s executor and wait for the reply."""
        ev = threading.Event()
        req = next(self._ids)
        slot = [ev, None, w]
        self._pending[req] = slot
        self._send(w, msg_fn(req))
        if not ev.wait(timeout):
            raise TimeoutError(f"request to worker {w} timed out")
        kind, payload = slot[1]
        if kind != "value":
            raise WorkerLost(f"worker {w} lost")
        return payload

    # ------------------------------------------------------------ user API

    def allocate(self, n: int, name: str = "") -> Slice:
        return self.rm.allocate(n, name)

    def compile(self, fn, *, name: str | None = None, jit: bool = True, n_out: int = 1) -> Program:
        return Program(fn, name or getattr(fn, "__name__", "program"), jit, n_out)

    def _ensure_compiled(self, prog: Program, workers: list, msgs: list) -> None:
        for w in workers:
            if w not in prog.compiled_on:
                prog.compiled_on.add(w)
                msgs.append((w, ("compile", prog.id, prog.payload, prog.jit, prog.name)))

    def put(self, value, slice_: Slice) -> Sharded:
        """Place host data on a slice. Pass a list to give each rank its own value."""
        workers = self.rm.physicals(slice_)
        values = value if isinstance(value, list) else [value] * len(workers)
        if "jax" in sys.modules:  # ship numpy, not device arrays
            import jax
            import numpy as np

            values = [jax.tree.map(np.asarray, v) for v in values]
        assert len(values) == len(workers)
        bufs, msgs = [], []
        for w, v in zip(workers, values):
            k = next(self._ids)
            msgs.append((w, ("put", k, v)))
            bufs.append(Buffer(self, w, k))
        self._dispatch(msgs)
        return Sharded(slice_, bufs)

    def run(self, prog: Program, slice_: Slice, *inputs: Sharded, consts=None, wait: bool = False):
        """Run ``prog`` once on every rank of ``slice_`` (an SPMD block).

        ``inputs`` are futures sharded over the same slice. ``consts`` is an
        optional list (one entry per rank) of tuples of extra scalar args, which
        is how a rank learns its index. Returns Sharded future(s) right away.
        """
        workers = self.rm.physicals(slice_)
        for x in inputs:
            for b, w in zip(x.buffers, workers):
                if b.wid != w:
                    raise ValueError(f"input {b} is not on worker {w}; transfer it first")
        epoch = self._epoch
        op_id = next(self._ids)
        msgs: list = []
        self._ensure_compiled(prog, workers, msgs)
        outs = [[] for _ in range(prog.n_out)]
        for r, w in enumerate(workers):
            keys = [next(self._ids) for _ in range(prog.n_out)]
            in_ids = [x.buffers[r].key for x in inputs]
            c = tuple(consts[r]) if consts is not None else ()
            msgs.append((w, ("run", epoch, op_id, prog.id, in_ids, c, keys, wait)))
            for j, k in enumerate(keys):
                outs[j].append(Buffer(self, w, k))
        self._dispatch_maybe_wait(op_id, len(workers), msgs, wait)
        res = [Sharded(slice_, o) for o in outs]
        return res[0] if prog.n_out == 1 else tuple(res)

    def allreduce(self, x: Sharded, reduce: str = "mean", wait: bool = False) -> Sharded:
        """Ring all-reduce across the slice ``x`` lives on, gang-scheduled."""
        workers = [b.wid for b in x.buffers]
        epoch, op_id = self._epoch, next(self._ids)
        msgs, bufs = [], []
        for b in x.buffers:
            k = next(self._ids)
            msgs.append((b.wid, ("allreduce", epoch, op_id, workers, b.key, k, reduce, wait)))
            bufs.append(Buffer(self, b.wid, k))
        self._dispatch_maybe_wait(op_id, len(workers), msgs, wait)
        return Sharded(x.slice, bufs)

    def _dispatch_maybe_wait(self, op_id, n, msgs, wait) -> None:
        if wait:
            entry = [n, threading.Event()]
            self._done[op_id] = entry
            self._dispatch(msgs)
            entry[1].wait()
            self._done.pop(op_id, None)
            self._check()
        else:
            self._dispatch(msgs)

    def transfer(self, x: Sharded, dst: Slice) -> Sharded:
        """Move a value onto another slice, worker to worker.

        Ranks map 1:1 when the sizes match. A single shard is broadcast to
        every rank of ``dst``. Buffers already on the right worker are reused
        without a copy.
        """
        dst_workers = self.rm.physicals(dst)
        if len(x) == len(dst_workers):
            srcs = x.buffers
        elif len(x) == 1:
            srcs = x.buffers * len(dst_workers)
        else:
            raise ValueError(f"cannot reshard {len(x)} shards onto {len(dst_workers)} devices")
        epoch = self._epoch
        msgs, bufs = [], []
        for b, w in zip(srcs, dst_workers):
            if b.wid == w:
                bufs.append(b)
                continue
            k = next(self._ids)
            msgs.append((b.wid, ("send", epoch, b.key, w, k)))
            bufs.append(Buffer(self, w, k))
        if msgs:
            self._dispatch(msgs)
        return Sharded(dst, bufs)

    def copy(self, buf: Buffer, dst_worker: int) -> Buffer:
        """Copy one buffer to a physical worker (used by recovery code)."""
        if buf.wid == dst_worker:
            return buf
        k = next(self._ids)
        self._dispatch([(buf.wid, ("send", self._epoch, buf.key, dst_worker, k))])
        return Buffer(self, dst_worker, k)

    def barrier(self, s: Slice | None = None, timeout: float = 120.0) -> None:
        """Wait until every worker in ``s`` (default: all alive) has drained its queue."""
        workers = self.rm.physicals(s) if s is not None else sorted(self.alive)
        for w in workers:
            self._request(w, lambda req: ("stats", req), timeout)

    # --------------------------------------------------- fault tolerance

    def kill(self, w: int) -> None:
        """SIGKILL a worker, as a hardware fault would."""
        os.kill(self.pids[w], signal.SIGKILL)

    def begin_recovery(self) -> list:
        """Acknowledge failures and re-open dispatch under a new epoch.

        Returns ``[(dead_worker, [virtual devices that were on it]), ...]``.
        """
        with self._sched_lock, self._state_lock:
            lost, self._lost = self._lost, []
            self._needs_recovery = False
            self._garbage = deque((w, k) for w, k in self._garbage if w in self.alive)
            return lost

    def held(self, w: int, keys: list) -> list:
        epoch = self._epoch
        return self._request(w, lambda req: ("query", epoch, req, keys))

    # ---------------------------------------------------------- tooling

    def set_trace(self, on: bool, t0: float | None = None) -> float:
        t0 = time.perf_counter() if t0 is None else t0
        for w in sorted(self.alive):
            self._send(w, ("config", {"trace": on, "t0": t0}))
        return t0

    def configure(self, **opts) -> None:
        for w in sorted(self.alive):
            self._send(w, ("config", opts))

    def traces(self) -> dict:
        return {w: self._request(w, lambda req: ("trace", req)) for w in sorted(self.alive)}

    def spmd(self, fn, slice_: Slice, args: tuple = (), timeout: float | None = None) -> list:
        """Multi-controller mode: every rank runs ``fn(ctx, *args)`` on its own.

        This is the baseline Pathways is compared against. The controller here
        only launches and watches, the way torchrun or mpirun would.
        """
        workers = self.rm.physicals(slice_)
        payload = cloudpickle.dumps(fn)
        self._spmd_results = {}
        self._spmd_event.clear()
        for r, w in enumerate(workers):
            self._send(w, ("spmd", payload, r, workers, args))
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(self._spmd_results) < len(workers):
            self._spmd_event.wait(0.1)
            self._spmd_event.clear()
            if self._needs_recovery:
                raise WorkerLost(f"lost workers {[w for w, _ in self._lost]}")
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("spmd job timed out")
        out = []
        for w in workers:
            kind, payload = self._spmd_results[w]
            if kind == "spmd_error":
                raise RemoteOpError(payload)
            out.append(payload)
        return out

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for w in list(self.alive):
            try:
                self._send(w, ("shutdown",))
            except OSError:
                pass
        for p in self.procs.values():
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
        self._listener.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
