"""The worker: one process per (simulated) accelerator host.

A worker owns a buffer store and a single in-order executor thread, the same
contract a TPU or GPU stream gives you: operations run in the order they were
enqueued, and an operation that needs a buffer that hasn't arrived yet blocks
the stream until it does. That one rule is why gang scheduling matters, since
two workers that see collectives in different orders will wait on each other
forever.

Threads in a worker:

* control reader: receives messages from the controller. Most go into the
  executor queue in order. ``abort``, ``fetch`` and ``send`` are handled
  immediately: a transfer waits on its source buffer, not on the stream.
* peer acceptor and one reader per peer: receive tensors and collective
  chunks from other workers and write them straight into the store or the
  mailbox, so the controller never sits on the data path.
* sender: pushes ``send`` (point-to-point transfer) payloads to peers so a
  transfer never stalls compute.
* executor: runs programs, collectives, and bookkeeping ops in order.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from collections import defaultdict
from multiprocessing.connection import Client, Listener


class Aborted(Exception):
    """Raised inside a blocking wait when the controller starts a new epoch."""


class RemoteError:
    """Placeholder stored in place of a buffer whose producing op failed."""

    def __init__(self, msg: str):
        self.msg = msg

    def __repr__(self) -> str:
        return f"RemoteError({self.msg.splitlines()[-1] if self.msg else ''})"


class Store:
    """Buffer store with futures semantics.

    ``get`` blocks until the buffer exists. ``on_ready`` registers a callback
    instead of blocking. ``delete`` on a buffer that hasn't arrived yet leaves
    a tombstone so the late arrival is dropped after its callbacks have run.
    """

    def __init__(self, worker: Worker):
        self._w = worker
        self._data: dict = {}
        self._waiters: dict = defaultdict(list)
        self._tomb: set = set()
        self.cv = threading.Condition()

    def put(self, key, value) -> None:
        with self.cv:
            cbs = self._waiters.pop(key, [])
            if key in self._tomb:
                self._tomb.discard(key)
            else:
                self._data[key] = value
            self.cv.notify_all()
        for cb in cbs:
            cb(value)

    def get(self, key, epoch: int, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cv:
            while key not in self._data:
                if epoch < self._w.epoch:
                    raise Aborted()
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"buffer {key} never arrived")
                self.cv.wait(0.05)
            return self._data[key]

    def on_ready(self, key, cb) -> None:
        with self.cv:
            if key in self._data:
                value = self._data[key]
            else:
                self._waiters[key].append(cb)
                return
        cb(value)

    def delete(self, key) -> None:
        with self.cv:
            if key in self._data:
                del self._data[key]
            elif key in self._waiters:
                self._tomb.add(key)

    def held(self, keys) -> list:
        with self.cv:
            return [k for k in keys if k in self._data and not isinstance(self._data[k], RemoteError)]

    def nbytes(self) -> int:
        with self.cv:
            return sum(getattr(v, "nbytes", 0) for v in self._data.values())


class Mailbox:
    """Keyed rendezvous for collective chunks."""

    def __init__(self, worker: Worker):
        self._w = worker
        self._box: dict = {}
        self.cv = threading.Condition()

    def put(self, key, value) -> None:
        with self.cv:
            self._box[key] = value
            self.cv.notify_all()

    def take(self, key, epoch: int, timeout: float):
        deadline = time.monotonic() + timeout
        with self.cv:
            while key not in self._box:
                if epoch < self._w.epoch:
                    raise Aborted()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"collective stalled waiting for {key!r}")
                self.cv.wait(0.05)
            return self._box.pop(key)

    def clear(self) -> None:
        with self.cv:
            self._box.clear()


def _to_host(x):
    """Convert a pytree of jax arrays to numpy so it can be pickled."""
    import jax
    import numpy as np

    return jax.tree.map(np.asarray, x)


class Worker:
    def __init__(self, wid: int, ctrl_addr, authkey: bytes):
        self.id = wid
        self.authkey = authkey
        self.epoch = 0
        self.store = Store(self)
        self.mailbox = Mailbox(self)
        self.programs: dict = {}
        self.execq: queue.Queue = queue.Queue()
        self.sendq: queue.Queue = queue.Queue()
        self.trace_on = False
        self.trace: list = []
        self.t0 = time.perf_counter()
        self.collective_timeout = 30.0

        self.peer_listener = Listener(family="AF_UNIX", authkey=authkey)
        self.peer_addrs: dict = {}
        self.peer_conns: dict = {}
        self.peer_locks: dict = defaultdict(threading.Lock)
        self.peer_conn_lock = threading.Lock()

        self.ctrl = Client(ctrl_addr, authkey=authkey)
        self.ctrl_lock = threading.Lock()
        self.send_ctrl(("hello", wid, self.peer_listener.address, os.getpid()))

    # -- plumbing ----------------------------------------------------------

    def send_ctrl(self, msg) -> None:
        with self.ctrl_lock:
            self.ctrl.send(msg)

    def _peer(self, wid):
        with self.peer_conn_lock:
            conn = self.peer_conns.get(wid)
            if conn is None:
                conn = Client(self.peer_addrs[wid], authkey=self.authkey)
                conn.send(("peer_hello", self.id))
                self.peer_conns[wid] = conn
            return conn

    def send_peer(self, wid, msg) -> None:
        conn = self._peer(wid)
        with self.peer_locks[wid]:
            conn.send(msg)

    def _accept_loop(self) -> None:
        while True:
            try:
                conn = self.peer_listener.accept()
            except OSError:
                return
            threading.Thread(target=self._peer_reader, args=(conn,), daemon=True).start()

    def _peer_reader(self, conn) -> None:
        try:
            conn.recv()  # peer_hello
            while True:
                kind, key, value = conn.recv()
                if kind == "data":
                    self.store.put(key, value)
                else:  # "coll"
                    self.mailbox.put(key, value)
        except (EOFError, OSError):
            return

    def _sender_loop(self) -> None:
        while True:
            dst, key, value = self.sendq.get()
            try:
                self.send_peer(dst, ("data", key, _to_host(value) if not isinstance(value, RemoteError) else value))
            except (OSError, EOFError, KeyError):
                pass  # peer is gone; the controller will notice and re-plan

    def _ctrl_loop(self) -> None:
        try:
            while True:
                msg = self.ctrl.recv()
                kind = msg[0]
                if kind == "abort":
                    new_epoch = msg[1]
                    self.epoch = max(self.epoch, new_epoch)
                    with self.store.cv:
                        self.store.cv.notify_all()
                    with self.mailbox.cv:
                        self.mailbox.cv.notify_all()
                    self.mailbox.clear()
                elif kind == "send":
                    # Transfers don't occupy the compute stream, like a DMA
                    # engine next to a GPU/TPU core. on_ready fires the moment
                    # the source buffer exists, even if compute queued behind it
                    # is still waiting. Deletes still follow in stream order, and
                    # this message arrives before any delete of the same buffer.
                    self._op_send(*msg[1:])
                elif kind == "fetch":
                    _, req, key = msg
                    self.store.on_ready(key, lambda v, req=req: self._reply_value(req, v))
                elif kind == "peers":
                    self.peer_addrs.update(msg[1])
                elif kind == "shutdown":
                    os._exit(0)
                else:
                    self.execq.put(msg)
        except (EOFError, OSError):
            os._exit(0)  # controller died: nothing left to do

    def _reply_value(self, req, v) -> None:
        if isinstance(v, RemoteError):
            self.send_ctrl(("value_error", req, v.msg))
        else:
            self.send_ctrl(("value", req, _to_host(v)))

    # -- executor ----------------------------------------------------------

    def _record(self, name, t0, t1, cat) -> None:
        if self.trace_on:
            self.trace.append((name, cat, t0 - self.t0, t1 - self.t0))

    def _exec_loop(self) -> None:
        while True:
            msg = self.execq.get()
            kind = msg[0]
            try:
                getattr(self, "_op_" + kind)(*msg[1:])
            except Exception:  # never let the stream die
                self.send_ctrl(("log", self.id, traceback.format_exc()))

    def _fail_outputs(self, out_ids, err: str) -> None:
        for o in out_ids:
            self.store.put(o, RemoteError(err))

    def _op_compile(self, prog_id, fn_bytes, use_jit, name) -> None:
        import cloudpickle

        fn = cloudpickle.loads(fn_bytes)
        if use_jit:
            import jax

            fn = jax.jit(fn)
        self.programs[prog_id] = (fn, name)

    def _op_put(self, key, value) -> None:
        self.store.put(key, value)

    def _op_delete(self, keys) -> None:
        for k in keys:
            self.store.delete(k)

    def _op_config(self, opts: dict) -> None:
        if "trace" in opts:
            self.trace_on = opts["trace"]
            self.trace = []
            self.t0 = opts.get("t0", time.perf_counter())
        if "collective_timeout" in opts:
            self.collective_timeout = opts["collective_timeout"]

    def _op_trace(self, req) -> None:
        events, self.trace = self.trace, []
        self.send_ctrl(("value", req, events))

    def _op_query(self, epoch, req, keys) -> None:
        self.send_ctrl(("value", req, self.store.held(keys)))

    def _op_stats(self, req) -> None:
        self.send_ctrl(("value", req, {"bytes": self.store.nbytes(), "pid": os.getpid()}))

    def _op_run(self, epoch, op_id, prog_id, in_ids, consts, out_ids, notify) -> None:
        if epoch < self.epoch:
            return self._fail_outputs(out_ids, "aborted")
        if prog_id not in self.programs:  # never hang on a protocol error
            err = f"program {prog_id} was never compiled on worker {self.id}"
            self.send_ctrl(("error", op_id, self.id, err))
            return self._fail_outputs(out_ids, err)
        fn, name = self.programs[prog_id]
        try:
            args = [self.store.get(i, epoch) for i in in_ids]
        except Aborted:
            return self._fail_outputs(out_ids, "aborted")
        bad = next((a for a in args if isinstance(a, RemoteError)), None)
        if bad is not None:
            return self._fail_outputs(out_ids, bad.msg)
        t0 = time.perf_counter()
        try:
            out = fn(*args, *consts)
            if len(out_ids) == 1:
                out = (out,)
            if self.trace_on or notify:
                import jax

                jax.block_until_ready(out)
        except Exception:
            err = traceback.format_exc()
            self.send_ctrl(("error", op_id, self.id, err))
            return self._fail_outputs(out_ids, err)
        t1 = time.perf_counter()
        self._record(name, t0, t1, "compute")
        for o, v in zip(out_ids, out):
            self.store.put(o, v)
        if notify:
            self.send_ctrl(("done", op_id, self.id))

    def _op_send(self, epoch, key, dst, dst_key) -> None:
        if epoch < self.epoch:
            return
        self.store.on_ready(key, lambda v: self.sendq.put((dst, dst_key, v)))

    def _op_allreduce(self, epoch, op_id, group, in_id, out_id, reduce, notify) -> None:
        if epoch < self.epoch:
            return self._fail_outputs([out_id], "aborted")
        t0 = time.perf_counter()
        try:
            x = self.store.get(in_id, epoch)
            if isinstance(x, RemoteError):
                return self._fail_outputs([out_id], x.msg)
            out = ring_allreduce(self, x, group, ("ar", op_id), epoch, reduce)
        except Aborted:
            return self._fail_outputs([out_id], "aborted")
        except Exception:
            err = traceback.format_exc()
            self.send_ctrl(("error", op_id, self.id, err))
            return self._fail_outputs([out_id], err)
        self._record("allreduce", t0, time.perf_counter(), "collective")
        self.store.put(out_id, out)
        if notify:
            self.send_ctrl(("done", op_id, self.id))

    def _op_spmd(self, fn_bytes, rank, group, args) -> None:
        """Multi-controller mode: run a whole user program locally.

        This is the SPMD baseline. Every worker runs the same Python loop, and
        the only coordination is through the collectives each one issues.
        """
        import cloudpickle

        fn = cloudpickle.loads(fn_bytes)
        ctx = SPMDContext(self, rank, group)
        try:
            result = fn(ctx, *args)
            self.send_ctrl(("spmd_done", self.id, _to_host(result)))
        except Exception:
            self.send_ctrl(("spmd_error", self.id, traceback.format_exc()))

    def serve(self) -> None:
        for target in (self._accept_loop, self._sender_loop, self._exec_loop):
            threading.Thread(target=target, daemon=True).start()
        self._ctrl_loop()


class SPMDContext:
    """What a multi-controller program sees: its rank and a collective."""

    def __init__(self, worker: Worker, rank: int, group: list):
        self.worker = worker
        self.rank = rank
        self.group = group
        self.world_size = len(group)
        self._n = 0

    def allreduce(self, x, reduce: str = "mean"):
        self._n += 1
        w = self.worker
        return ring_allreduce(w, x, self.group, ("spmd", self._n), w.epoch, reduce)

    def log(self, **kv) -> None:
        self.worker.send_ctrl(("spmd_log", self.worker.id, self.rank, kv))


def ring_allreduce(worker: Worker, x, group: list, tag, epoch: int, reduce: str):
    """Bandwidth-optimal ring all-reduce (reduce-scatter, then all-gather).

    Works on any pytree of float arrays. Each rank sends 2*(n-1)/n of the
    data in total, no matter how many ranks there are. The final chunk values
    are computed once and broadcast in the all-gather, so every rank ends up
    with bit-identical results. Data-parallel replicas therefore stay exactly
    in sync, which recovery relies on.
    """
    import jax
    import numpy as np

    leaves, treedef = jax.tree.flatten(x)
    shapes = [np.shape(leaf) for leaf in leaves]
    flat = np.concatenate([np.asarray(leaf, dtype=np.float32).ravel() for leaf in leaves])
    n = len(group)
    if n > 1:
        r = group.index(worker.id)
        nxt = group[(r + 1) % n]
        chunks = np.array_split(flat, n)
        timeout = worker.collective_timeout
        for s in range(n - 1):  # reduce-scatter
            i_send, i_recv = (r - s) % n, (r - s - 1) % n
            worker.send_peer(nxt, ("coll", (tag, "rs", s, i_send), chunks[i_send]))
            chunks[i_recv] = chunks[i_recv] + worker.mailbox.take((tag, "rs", s, i_recv), epoch, timeout)
        for s in range(n - 1):  # all-gather
            i_send, i_recv = (r - s + 1) % n, (r - s) % n
            worker.send_peer(nxt, ("coll", (tag, "ag", s, i_send), chunks[i_send]))
            chunks[i_recv] = worker.mailbox.take((tag, "ag", s, i_recv), epoch, timeout)
        flat = np.concatenate(chunks)
    if reduce == "mean":
        flat = flat / n
    out, off = [], 0
    for shp in shapes:
        size = int(np.prod(shp)) if shp else 1
        out.append(flat[off : off + size].reshape(shp))
        off += size
    return jax.tree.unflatten(treedef, out)


def worker_main(wid: int, ctrl_addr, authkey: bytes, env: dict) -> None:
    os.environ.update(env)
    # Import heavy modules once, before any thread starts. Concurrent first
    # imports from several threads can trip CPython's import-lock deadlock check.
    import cloudpickle  # noqa: F401
    import jax  # noqa: F401
    import numpy  # noqa: F401

    Worker(wid, ctrl_addr, authkey).serve()
