"""E4: kill a worker mid-training. What does each design lose?

Workload: data-parallel training of the 0.8M-param char transformer on Tiny
Shakespeare, 4 replicas, global batch 128 x 64 tokens.

* SPMD (multi-controller), the usual DDP shape: 4 processes run the same
  loop and rank 0 checkpoints to disk every ``CKPT_EVERY`` steps. When a
  process dies, its peers stall in the all-reduce. The launcher (think
  torchrun) tears the job down, starts 4 fresh processes, and reloads the
  last checkpoint. Detection is made instant here, which is generous to SPMD.

* Single controller (Pathways-style): the controller sees the failure,
  aborts in-flight work on survivors, and maps the lost virtual device onto
  a hot spare. It restores state from a surviving replica's in-memory
  buffers over worker-to-worker transfers, then re-dispatches from the last
  step a survivor completed. A second kill with no spares left shrinks the
  job to 3 replicas (elastic) and keeps going.

Since batches are a pure function of (step, rank) and the ring all-reduce is
deterministic, the recovered single-controller run should match a run with
no failure *bit for bit* until the elastic shrink. The script checks this.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections import deque

import numpy as np

from experiments.common import save
from minipath import Cluster, RemoteOpError, Sharded, WorkerLost
from minipath.workloads import shakespeare
from minipath.workloads import transformer as T

N = 4
CKPT_EVERY = 50
LAG = 2  # steps the controller is allowed to run ahead of observed losses
KEEP = LAG + 3  # steps of optimizer state kept alive for recovery


class DPTrainer:
    """Data-parallel training, driven by one controller."""

    def __init__(self, c: Cluster, cfg: T.Config, tokens: np.ndarray, n: int = N):
        self.c, self.cfg = c, cfg
        self.s = c.allocate(n, "dp")
        progs = T.make_programs(cfg)
        self.batch = c.compile(T.make_batcher(cfg), name="batch", n_out=2)
        self.lg = c.compile(progs["loss_and_grad"], name="loss_and_grad", n_out=2)
        self.adam = c.compile(progs["adam"], name="adam", n_out=4)
        params = T.init(0, cfg)[0]
        z = T.zeros_like(params)
        self.state = {0: tuple(c.put(v, self.s) for v in (params, z, z, np.float32(0)))}
        self.tok = {b.wid: b for b in c.put(tokens, self.s).buffers}
        self.events: list = []

    def tokens(self) -> Sharded:
        return Sharded(self.s, [self.tok[w] for w in self.c.rm.physicals(self.s)])

    def step(self, k: int):
        c, s = self.c, self.s
        p, m, v, t = self.state[k]
        x, y = c.run(self.batch, s, self.tokens(), consts=[(np.int32(k), np.int32(r)) for r in range(len(s))])
        loss, g = c.run(self.lg, s, p, x, y)
        g = c.allreduce(g)
        self.state[k + 1] = c.run(self.adam, s, p, m, v, g, t)
        for old in [j for j in self.state if j < k + 1 - KEEP]:
            del self.state[old]
        return loss

    def recover(self) -> int:
        """Re-plan after a failure. Returns the step to resume from."""
        c = self.c
        t0 = time.perf_counter()
        lost = c.begin_recovery()
        old_workers = {v: c.rm.physical(v) for v in self.s.vdevs}
        for dead, vdevs in lost:
            for v in vdevs:
                new = c.rm.remap(v)
                self.events.append(
                    {
                        "t": time.perf_counter(),
                        "kind": "remap" if new is not None else "no_spare",
                        "dead": dead,
                        "vdev": v,
                        "new_worker": new,
                    }
                )
        if any(c.rm.physical(v) in c.rm.dead for v in self.s.vdevs):
            before = len(self.s)
            self.s = c.rm.shrink(self.s)
            self.events.append({"t": time.perf_counter(), "kind": "shrink", "from": before, "to": len(self.s)})

        # Find the newest step for which some survivor holds the full state.
        resume, src = None, None
        for k in sorted(self.state, reverse=True):
            for r, v in enumerate(old_workers):
                w = old_workers[v]
                if w not in c.alive:
                    continue
                keys = [x.buffers[r].key for x in self.state[k]]
                if len(c.held(w, keys)) == len(keys):
                    resume, src = k, (r, w)
                    break
            if resume is not None:
                break
        if resume is None:
            raise RuntimeError("no surviving replica holds a complete state")

        r_src, w_src = src
        old = self.state[resume]
        src_bufs = [x.buffers[r_src] for x in old]
        cols = []  # per new rank: the 4 state buffers
        for v in self.s.vdevs:
            w = c.rm.physical(v)
            r_old = list(old_workers).index(v)
            mine = [x.buffers[r_old] for x in old]
            if old_workers[v] == w and len(c.held(w, [b.key for b in mine])) == len(mine):
                cols.append(mine)  # survivor already has it: no copy
            else:
                cols.append([c.copy(b, w) for b in src_bufs])
        new_state = [Sharded(self.s, [col[j] for col in cols]) for j in range(len(old))]
        self.state = {resume: tuple(new_state)}
        any_tok = next(b for w, b in self.tok.items() if w in c.alive)
        for w in c.rm.physicals(self.s):
            if w not in self.tok or w not in c.alive:
                self.tok[w] = c.copy(any_tok, w)
        self.events.append(
            {
                "t": time.perf_counter(),
                "kind": "restored",
                "resume_step": resume,
                "source_worker": w_src,
                "replan_s": time.perf_counter() - t0,
            }
        )
        return resume

    def train(self, total: int, kill_at: dict | None = None, log: list | None = None) -> list:
        """Run to ``total`` steps. ``kill_at[step] = worker`` kills when that step's loss is seen."""
        kill_at = dict(kill_at or {})
        log = [] if log is None else log
        k, inflight = 0, deque()
        while k < total or inflight:
            try:
                if k < total and len(inflight) < LAG:
                    inflight.append((k, self.step(k)))
                    k += 1
                    continue
                kk, lf = inflight.popleft()
                loss = float(lf.fetch(0))
                log.append({"step": kk, "loss": loss, "t": time.perf_counter(), "replicas": len(self.s)})
                if kk in kill_at:
                    w = kill_at.pop(kk)
                    self.events.append({"t": time.perf_counter(), "kind": "kill", "worker": w, "after_step": kk})
                    self.c.kill(w)
            except (WorkerLost, RemoteOpError) as e:
                if not self.c._needs_recovery:
                    time.sleep(0.05)  # the reader thread may not have noticed yet
                    if not self.c._needs_recovery:
                        raise e
                self.events.append({"t": time.perf_counter(), "kind": "detected"})
                k = self.recover()
                inflight.clear()
        return log

    def params(self):
        return self.state[max(self.state)][0].fetch(0)


# ------------------------------------------------------------- SPMD baseline


def spmd_train(ctx, cfg, total, ckpt_dir, ckpt_every):
    """The multi-controller program. Every rank runs this same function."""
    import jax
    import numpy as np_

    from minipath.workloads import shakespeare as sh
    from minipath.workloads import transformer as T_

    tokens = sh.load()[0]
    progs = T_.make_programs(cfg)
    batch = jax.jit(T_.make_batcher(cfg))
    lg = jax.jit(progs["loss_and_grad"])
    adam = jax.jit(progs["adam"])
    ckpt = os.path.join(ckpt_dir, "ckpt.pkl")
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            start, (p, m, v, t) = pickle.load(f)
    else:
        p = T_.init(0, cfg)[0]
        m = v = T_.zeros_like(p)
        t, start = np_.float32(0), 0
    ctx.log(event="ready", step=start)
    for k in range(start, total):
        x, y = batch(tokens, np_.int32(k), np_.int32(ctx.rank))
        loss, g = lg(p, x, y)
        g = ctx.allreduce(g)
        p, m, v, t = adam(p, m, v, g, t)
        if ctx.rank == 0:
            ctx.log(step=k, loss=float(loss))
            if (k + 1) % ckpt_every == 0:
                tmp = ckpt + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump((k + 1, jax.tree.map(np_.asarray, (p, m, v, t))), f)
                os.replace(tmp, ckpt)
                ctx.log(event="ckpt", step=k + 1)
    return start


def run_spmd(cfg, total: int, kill_after: int | None, kill_rank: int = 2) -> dict:
    ckpt_dir = tempfile.mkdtemp(prefix="minipath_ckpt_")
    log, events, attempt = [], [], 0
    t_begin = time.perf_counter()
    killed = kill_after is None
    try:
        while True:
            attempt += 1
            t_launch = time.perf_counter()
            c = Cluster(N)
            s = c.allocate(N)
            events.append({"t": t_launch, "kind": "launch", "attempt": attempt})
            stop = threading.Event()

            def watch(c=c):
                seen = 0
                while not stop.is_set():
                    while seen < len(c.spmd_logs):
                        ts, _, rank, kv = c.spmd_logs[seen]
                        seen += 1
                        if "loss" in kv:
                            log.append({"step": kv["step"], "loss": kv["loss"], "t": ts, "attempt": attempt})
                        else:
                            events.append({"t": ts, "kind": kv["event"], "step": kv["step"], "rank": rank})
                        nonlocal killed
                        if not killed and kv.get("step") == kill_after and "loss" in kv:
                            killed = True
                            events.append(
                                {
                                    "t": time.perf_counter(),
                                    "kind": "kill",
                                    "worker": kill_rank,
                                    "after_step": kill_after,
                                }
                            )
                            c.kill(kill_rank)
                    time.sleep(0.002)

            th = threading.Thread(target=watch, daemon=True)
            th.start()
            try:
                c.spmd(spmd_train, s, (cfg, total, ckpt_dir, CKPT_EVERY))
                stop.set()
                th.join()
                c.shutdown()
                break
            except WorkerLost:
                events.append({"t": time.perf_counter(), "kind": "detected"})
                stop.set()
                th.join()
                c.shutdown()  # tear down the whole job, like torchrun does
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    return {"log": log, "events": events, "wall": time.perf_counter() - t_begin, "t0": t_begin}


def run_single_controller(cfg, tokens, total: int, kill_at: dict | None, spares: int) -> dict:
    t_begin = time.perf_counter()
    with Cluster(N, n_spares=spares) as c:
        tr = DPTrainer(c, cfg, tokens)
        log = tr.train(total, kill_at)
        final = tr.params()
        rm_events = c.rm.events
    return {
        "log": log,
        "events": tr.events,
        "rm_events": [list(map(str, e)) for e in rm_events],
        "wall": time.perf_counter() - t_begin,
        "t0": t_begin,
        "final_params": final,
    }


def summarize(run: dict, total: int) -> dict:
    log = run["log"]
    steps = [r["step"] for r in log]
    executed = len(steps)
    kill_ts = [e["t"] for e in run["events"] if e["kind"] == "kill"]
    out = {"wall_s": run["wall"], "steps_executed": executed, "steps_redone": executed - total}
    downs = []
    for kt in kill_ts:
        before = max((r["step"] for r in log if r["t"] <= kt), default=-1)
        after = next((r["t"] for r in log if r["t"] > kt and r["step"] > before), None)
        if after is not None:
            downs.append(after - kt)
    out["time_to_new_progress_s"] = downs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = T.Config()
    train, val, vocab = shakespeare.load()
    total = 120 if args.quick else 400
    k1, k2 = (40, 90) if args.quick else (130, 280)

    print("clean single-controller run ...")
    clean = run_single_controller(cfg, train, total, None, spares=0)
    print(f"  {clean['wall']:.1f}s, final loss {clean['log'][-1]['loss']:.4f}")

    print(
        f"single controller, kill worker 2 after step {k1} (hot spare) and worker 1 after step {k2} (no spare left) ..."
    )
    sc = run_single_controller(cfg, train, total, {k1: 2, k2: 1}, spares=1)
    print(f"  {sc['wall']:.1f}s, final loss {sc['log'][-1]['loss']:.4f}")

    print(f"SPMD + checkpoint every {CKPT_EVERY}, kill rank 2 after step {k1} ...")
    sp = run_spmd(cfg, total, k1)
    print(f"  {sp['wall']:.1f}s, final loss {sp['log'][-1]['loss']:.4f}")

    print("SPMD clean ...")
    sp_clean = run_spmd(cfg, total, None)
    print(f"  {sp_clean['wall']:.1f}s")

    # Bit-exactness: compare per-step losses up to the elastic shrink.
    clean_loss = {r["step"]: r["loss"] for r in clean["log"]}
    sc_last = {}
    for r in sc["log"]:
        sc_last[r["step"]] = r  # last occurrence wins (re-executed steps)
    pre_shrink = [s for s, r in sc_last.items() if r["replicas"] == N]
    exact = all(sc_last[s]["loss"] == clean_loss[s] for s in pre_shrink)
    print(f"recovered run bit-identical to clean run for steps 0..{max(pre_shrink)}: {exact}")

    # Held-out loss of the final models
    import jax

    progs = T.make_programs(cfg)
    ev = jax.jit(progs["eval_loss"])
    vb = jax.jit(T.make_batcher(cfg, 256))
    xs = [vb(val, np.int32(i), np.int32(0)) for i in range(4)]
    val_clean = float(np.mean([ev(clean["final_params"], x, y) for x, y in xs]))
    val_sc = float(np.mean([ev(sc["final_params"], x, y) for x, y in xs]))
    sample = T.generate(clean["final_params"], cfg, vocab, "ROMEO:", 300)
    print(f"val loss clean={val_clean:.4f} recovered+elastic={val_sc:.4f}")
    print(sample)

    def strip(run):
        return {k: v for k, v in run.items() if k != "final_params"}

    save(
        "e4_fault_tolerance",
        {
            "config": cfg.__dict__,
            "replicas": N,
            "total_steps": total,
            "ckpt_every": CKPT_EVERY,
            "kills": {"first_after_step": k1, "second_after_step": k2},
            "bit_identical_until_shrink": exact,
            "val_loss": {"clean": val_clean, "recovered_elastic": val_sc},
            "sample": sample,
            "summary": {
                "single_controller": summarize(sc, total),
                "spmd": summarize(sp, total),
                "clean_single_controller": summarize(clean, total),
                "clean_spmd": summarize(sp_clean, total),
            },
            "runs": {"single_controller": strip(sc), "spmd": sp, "clean": strip(clean), "clean_spmd": sp_clean},
        },
    )


if __name__ == "__main__":
    main()
