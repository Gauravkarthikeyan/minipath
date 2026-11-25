"""E3: MPMD pipeline parallelism driven by one controller.

A 4-layer transformer is cut into 4 stages on 4 workers. Each stage runs
different programs: the first embeds tokens, the last fuses the head, loss
and first backward step. The controller emits the GPipe schedule as a plain
Python loop over futures, and the workers' in-order streams plus data
dependencies produce the pipeline on their own. Nothing is hand-synchronized.

Measured:
  1. Correctness: the pipelined loss curve matches single-device training
     with the same init and batches (to float32 noise).
  2. The bubble: idle fraction from worker traces vs. the GPipe bound
     (P-1)/(M+P-1), for M micro-batches.
  3. Throughput in tokens/s against one worker running the whole model.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.common import save
from minipath import Cluster, Sharded, Slice
from minipath.workloads import shakespeare
from minipath.workloads import transformer as T

P = 4
GLOBAL_BATCH = 32


class Pipeline:
    def __init__(self, c: Cluster, cfg: T.Config, n_micro: int, train_tokens, seed: int = 0):
        self.c, self.cfg, self.M = c, cfg, n_micro
        self.stages = [c.allocate(1, f"stage{i}") for i in range(P)]
        progs = T.make_programs(cfg, P, n_micro)
        self.fwd = c.compile(progs["fwd"], name="fwd")
        self.bwd = c.compile(progs["bwd"], name="bwd", n_out=2)
        self.bwd_first = c.compile(progs["bwd_first"], name="bwd")
        self.last = c.compile(progs["last"], name="fwd+bwd", n_out=3)
        self.add = c.compile(progs["add"], name="accum")
        self.adam = c.compile(progs["adam_global"], name="adam", n_out=4)
        self.sqnorm = c.compile(progs["sqnorm"], name="norm")
        self.all_stages = Slice(tuple(v for s in self.stages for v in s.vdevs), "all")
        self.batch = c.compile(T.make_batcher(cfg, GLOBAL_BATCH // n_micro), name="batch", n_out=2)
        params = T.init(seed, cfg, P)
        self.p = [c.put(params[i], s) for i, s in enumerate(self.stages)]
        self.m = [c.put(T.zeros_like(params[i]), s) for i, s in enumerate(self.stages)]
        self.v = [c.put(T.zeros_like(params[i]), s) for i, s in enumerate(self.stages)]
        self.t = [c.put(np.float32(0), s) for s in self.stages]
        self.tok_first = c.put(train_tokens, self.stages[0])
        self.tok_last = c.put(train_tokens, self.stages[-1])

    def step(self, step: int):
        """Dispatch one GPipe step. Returns the per-micro-batch loss futures."""
        c, S, M = self.c, self.stages, self.M
        acts = [[None] * M for _ in range(P)]  # input activation to stage i for micro m
        grads = [None] * P
        losses = []
        gx_next = [None] * M
        # forward wave (the last stage fuses loss + its backward)
        for m in range(M):
            consts = [(np.int32(step), np.int32(m))]
            x, _ = c.run(self.batch, S[0], self.tok_first, consts=consts)
            _, y = c.run(self.batch, S[-1], self.tok_last, consts=consts)
            h = x
            for i in range(P - 1):
                acts[i][m] = h
                h = c.transfer(c.run(self.fwd, S[i], self.p[i], h), S[i + 1])
            acts[P - 1][m] = h
            loss, gp, gx = c.run(self.last, S[-1], self.p[-1], h, y)
            losses.append(loss)
            grads[-1] = gp if grads[-1] is None else c.run(self.add, S[-1], grads[-1], gp)
            gx_next[m] = c.transfer(gx, S[-2])
        # backward wave, stage P-2 down to 0
        for i in range(P - 2, -1, -1):
            for m in range(M):
                if i == 0:
                    gp = c.run(self.bwd_first, S[0], self.p[0], acts[0][m], gx_next[m])
                else:
                    gp, gx = c.run(self.bwd, S[i], self.p[i], acts[i][m], gx_next[m])
                    gx_next[m] = c.transfer(gx, S[i - 1])
                grads[i] = gp if grads[i] is None else c.run(self.add, S[i], grads[i], gp)
        # Global grad norm for clipping: one all-reduce across all four
        # stage workers, even though each runs a different program.
        sq = [c.run(self.sqnorm, S[i], grads[i]) for i in range(P)]
        sq = c.allreduce(Sharded(self.all_stages, [x.buffers[0] for x in sq]), reduce="sum")
        for i in range(P):
            self.p[i], self.m[i], self.v[i], self.t[i] = c.run(
                self.adam, S[i], self.p[i], self.m[i], self.v[i], grads[i], self.t[i], sq[i]
            )
        return losses


def single_device_run(c: Cluster, cfg: T.Config, n_micro: int, tokens, steps: int, seed: int = 0):
    """Reference: the whole model on one worker, same micro-batches, accumulated grads."""
    s = c.allocate(1, "single")
    progs = T.make_programs(cfg, 1, n_micro)
    lg = c.compile(progs["loss_and_grad"], n_out=2)
    add = c.compile(progs["add"])
    adam = c.compile(progs["adam"], n_out=4)
    batch = c.compile(T.make_batcher(cfg, GLOBAL_BATCH // n_micro), n_out=2)
    params = T.init(seed, cfg, 1)[0]
    p, m, v, t = (
        c.put(params, s),
        c.put(T.zeros_like(params), s),
        c.put(T.zeros_like(params), s),
        c.put(np.float32(0), s),
    )
    tok = c.put(tokens, s)
    curve, t0 = [], None
    for k in range(steps):
        if k == 3:
            c.barrier(s)
            t0 = time.perf_counter()
        g, ls = None, []
        for mi in range(n_micro):
            x, y = c.run(batch, s, tok, consts=[(np.int32(k), np.int32(mi))])
            loss, gm = c.run(lg, s, p, x, y)
            ls.append(loss)
            g = gm if g is None else c.run(add, s, g, gm)
        p, m, v, t = c.run(adam, s, p, m, v, g, t)
        curve.append(ls)
    curve = [float(np.mean([float(f.fetch(0)) * n_micro for f in ls])) for ls in curve]
    dt = (time.perf_counter() - t0) / (steps - 3)
    c.rm.release(s)
    return curve, dt


def bubble_from_trace(events: dict, t_start: float, t_end: float) -> dict:
    busy = {}
    for w, evs in events.items():
        b = sum(min(e[3], t_end) - max(e[2], t_start) for e in evs if e[3] > t_start and e[2] < t_end)
        busy[w] = b
    span = t_end - t_start
    util = float(np.mean([b / span for b in busy.values()]))
    return {"utilization": util, "bubble": 1 - util, "per_worker_util": {str(w): b / span for w, b in busy.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = T.Config()
    train, val, vocab = shakespeare.load()
    curve_steps = 40 if args.quick else 150
    out: dict = {"config": cfg.__dict__, "P": P, "global_batch": GLOBAL_BATCH}

    with Cluster(P) as c:
        # 1. correctness: pipeline(M=4) vs single device (same micro-batches)
        pipe = Pipeline(c, cfg, 4, train)
        pipe_losses = []
        for k in range(curve_steps):
            pipe_losses.append(pipe.step(k))
        pipe_curve = [float(np.mean([float(f.fetch(0)) for f in ls])) for ls in pipe_losses]
        for s in pipe.stages:
            c.rm.release(s)
        del pipe
        single_curve, single_dt = single_device_run(c, cfg, 4, train, curve_steps)
        diff = float(np.max(np.abs(np.array(pipe_curve) - np.array(single_curve))))
        print(
            f"loss after {curve_steps} steps: pipeline={pipe_curve[-1]:.4f} single={single_curve[-1]:.4f} max|diff|={diff:.2e}"
        )
        out["equivalence"] = {"pipeline": pipe_curve, "single": single_curve, "max_abs_diff": diff}

        # single-device throughput with the full batch in one micro-batch
        _, single_dt_m1 = single_device_run(c, cfg, 1, train, 12)
        out["single_device"] = {"sec_per_step": single_dt_m1, "tokens_per_sec": GLOBAL_BATCH * cfg.ctx / single_dt_m1}
        print(f"single device: {single_dt_m1 * 1e3:.0f} ms/step")

        # 2+3. bubble and throughput vs micro-batches
        sweep = []
        for M in (1, 2, 4, 8, 16):
            pipe = Pipeline(c, cfg, M, train)
            for k in range(3):  # compile + warm up
                pipe.step(k)
            c.barrier()
            n_meas = 4 if args.quick else 8
            t0 = c.set_trace(True)
            t_start = time.perf_counter()
            for k in range(3, 3 + n_meas):
                ls = pipe.step(k)
            ls[-1].fetch(0)
            c.barrier()
            t_end = time.perf_counter()
            tr = c.traces()
            c.set_trace(False)
            dt = (t_end - t_start) / n_meas
            b = bubble_from_trace(tr, 0.0, t_end - t0)
            theory = (P - 1) / (M + P - 1)
            row = {
                "M": M,
                "sec_per_step": dt,
                "tokens_per_sec": GLOBAL_BATCH * cfg.ctx / dt,
                "speedup_vs_single": single_dt_m1 / dt,
                "bubble_measured": b["bubble"],
                "bubble_theory": theory,
                "per_worker_util": b["per_worker_util"],
            }
            if M in (4, 8):  # keep one step of trace for the timeline figure
                last_step = [(w, e) for w, evs in tr.items() for e in evs]
                row["trace"] = {str(w): [list(e) for e in evs] for w, evs in tr.items()}
                row["trace_window"] = [0.0, t_end - t0]
                del last_step
            sweep.append(row)
            print(
                f"M={M:2d}: {dt * 1e3:6.0f} ms/step  bubble measured={b['bubble']:.2f} theory={theory:.2f}  speedup={row['speedup_vs_single']:.2f}x"
            )
            for s in pipe.stages:
                c.rm.release(s)
            del pipe
        out["sweep"] = sweep

    save("e3_pipeline", out)


if __name__ == "__main__":
    main()
