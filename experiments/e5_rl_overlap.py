"""E5: RL with heterogeneous programs on separate slices, synchronous vs overlapped.

This is the RL post-training shape from the blog: a sampler program
(generation) and a learner program (gradient step) run on different
devices, and weights flow from learner to sampler after each update.

* sync:  sample with theta_k, train on it, push theta_{k+1}, sample again.
         The sampler slice idles while the learner trains, and vice versa.
* async: the rollout for iteration k+1 is dispatched *before* update k,
         using theta_k. Sampling and training overlap and the policy lags by
         one step (one-step off-policy, as in many RL systems).

With a single controller the two schedules differ only in where one line
of Python sits. Measured: throughput, how busy each slice is (from worker
traces), and CartPole return as a function of iteration and of wall-clock.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.common import save
from minipath import Cluster
from minipath.workloads import cartpole as CP


def run(mode: str, iters: int, seed: int, cfg: CP.RLConfig, trace_window: tuple | None = None) -> dict:
    with Cluster(4) as c:
        S = c.allocate(2, "sampler")
        L = c.allocate(2, "learner")
        progs = CP.make_programs(cfg)
        rollout = c.compile(progs["rollout"], name="rollout", n_out=5)
        lg = c.compile(progs["loss_and_grad"], name="learn", n_out=2)
        adam = c.compile(progs["adam"], name="adam", n_out=4)
        params = CP.init_policy(seed, cfg)
        z = {k: np.zeros(v.shape, np.float32) for k, v in params.items()}
        p, m, v, t = c.put(params, L), c.put(z, L), c.put(z, L), c.put(np.float32(0), L)

        def sample(sp, k):
            return c.run(rollout, S, sp, consts=[(np.int32(k + 1000 * seed), np.int32(r)) for r in range(2)])

        sp = c.transfer(p[0], S)
        batch = sample(sp, 0)
        log, pending, t0, trace = [], [], None, None
        for k in range(iters):
            if k == 3:  # skip compile warm-up
                c.barrier()
                t0 = time.perf_counter()
            if trace_window and k == trace_window[0]:
                c.barrier()
                t_tr = c.set_trace(True)
            nxt = sample(sp, k + 1) if mode == "async" else None
            data = [c.transfer(x, L) for x in batch[:4]]
            _, g = c.run(lg, L, p, *data, consts=[(np.int32(k * 2 + r + 100_000 * seed),) for r in range(2)])
            g = c.allreduce(g)
            p, m, v, t = c.run(adam, L, p, m, v, g, t)
            sp = c.transfer(p[0], S)
            if mode == "sync":
                nxt = sample(sp, k + 1)
            pending.append((k, batch[4]))
            if len(pending) > 2:
                kk, ep = pending.pop(0)
                log.append({"iter": kk, "return": float(np.mean(ep.fetch())), "t": time.perf_counter()})
            if trace_window and k == trace_window[1]:
                c.barrier()
                trace = {"events": c.traces(), "t_end": time.perf_counter() - t_tr}
                c.set_trace(False)
            batch = nxt
        for kk, ep in pending:
            log.append({"iter": kk, "return": float(np.mean(ep.fetch())), "t": time.perf_counter()})
        c.barrier()
        wall = time.perf_counter() - t0
        sampler_w, learner_w = c.rm.physicals(S), c.rm.physicals(L)
    for r in log:
        r["t"] -= t0
    steps = 2 * cfg.envs * cfg.horizon
    out = {
        "mode": mode,
        "seed": seed,
        "log": log,
        "sec_per_iter": wall / (iters - 3),
        "env_steps_per_sec": steps * (iters - 3) / wall,
        "sampler_workers": sampler_w,
        "learner_workers": learner_w,
    }
    if trace:
        span = trace["t_end"]
        busy = {w: sum(e[3] - e[2] for e in evs) / span for w, evs in trace["events"].items()}
        out["util"] = {
            "sampler": float(np.mean([busy[w] for w in sampler_w])),
            "learner": float(np.mean([busy[w] for w in learner_w])),
        }
        out["trace"] = {str(w): [list(e) for e in evs] for w, evs in trace["events"].items()}
        out["trace_span"] = span
    return out


def first_reaching(log, thresh):
    for r in log:
        if r["return"] >= thresh:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = CP.RLConfig()
    iters = 40 if args.quick else 120
    seeds = (0,) if args.quick else (0, 1, 2)
    thresh = 195.0
    runs = []
    for seed in seeds:
        for mode in ("sync", "async"):
            r = run(mode, iters, seed, cfg, trace_window=(10, 14) if seed == 0 else None)
            hit = first_reaching(r["log"], thresh)
            r["reached"] = None if hit is None else {"iter": hit["iter"], "t": hit["t"]}
            runs.append(r)
            util = r.get("util", {})
            print(
                f"seed={seed} {mode:5s}: {r['sec_per_iter'] * 1e3:5.0f} ms/iter  {r['env_steps_per_sec'] / 1e3:6.0f}k env-steps/s  "
                f"final return {r['log'][-1]['return']:.1f}  reached {thresh:.0f}: {r['reached']}  util {util}"
            )
    save("e5_rl_overlap", {"config": cfg.__dict__, "iters": iters, "threshold": thresh, "runs": runs})


if __name__ == "__main__":
    main()
