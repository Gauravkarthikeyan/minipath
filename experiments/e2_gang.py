"""E2: why Pathways needs a central gang scheduler.

Two clients share the same 4 workers, and each one runs a chain of
all-reduces. Every worker executes ops in arrival order, the same contract a
TPU or GPU stream gives you. If worker 0 sees client A's all-reduce first
while worker 1 sees client B's first, each waits in a collective the other
will never join. That is a deadlock.

* unscheduled_random: each client sends its op to the members on its own,
  in no particular order (independent clients, different network paths).
* unscheduled_fixed: same, but every client always sends to members in
  rank order. The only nondeterminism left is the thread race between
  the two clients, which is the best case without a scheduler.
* with gang scheduling, one sequencer delivers each op to its whole gang
  atomically, so every worker sees the same order.

A trial counts as a deadlock when any collective stalls longer than the
timeout. That is measured, not assumed.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from experiments.common import save
from minipath import Cluster, RemoteOpError


def add_one(x):
    return x + 1.0


def client_loop(c: Cluster, s, prog, n_ops: int, result: dict, name: str) -> None:
    try:
        x = c.put(np.zeros((256,), np.float32), s)
        for _ in range(n_ops):
            x = c.allreduce(c.run(prog, s, x))
        v = x.fetch(0, timeout=30)
        result[name] = ("ok", float(v[0]))
    except (RemoteOpError, TimeoutError) as e:
        result[name] = ("stalled", type(e).__name__)


def trial(mode: str, n_ops: int, timeout: float) -> dict:
    gang = mode == "gang_scheduled"
    with Cluster(4, gang_scheduling=gang, shuffle_unscheduled=mode == "unscheduled_random") as c:
        c.configure(collective_timeout=timeout)
        s = c.allocate(4)
        prog = c.compile(add_one)
        # warm up the compile so both clients start racing at once
        c.allreduce(c.run(prog, s, c.put(np.zeros((256,), np.float32), s))).fetch(0)
        result: dict = {}
        threads = [threading.Thread(target=client_loop, args=(c, s, prog, n_ops, result, n)) for n in ("A", "B")]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.perf_counter() - t0
    ok = all(r[0] == "ok" for r in result.values())
    correct = ok and all(abs(r[1] - n_ops) < 1e-3 for r in result.values())
    return {"deadlock": not ok, "correct": correct, "seconds": dt, "result": result}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_trials = 6 if args.quick else 20
    n_ops = 50
    timeout = 2.0
    out = {"n_trials": n_trials, "ops_per_client": n_ops, "stall_timeout_s": timeout, "runs": {}}
    for key in ("unscheduled_random", "unscheduled_fixed", "gang_scheduled"):
        runs = [trial(key, n_ops, timeout) for _ in range(n_trials)]
        dl = sum(r["deadlock"] for r in runs)
        good = [r["seconds"] for r in runs if not r["deadlock"]]
        out["runs"][key] = {
            "deadlocks": dl,
            "deadlock_rate": dl / n_trials,
            "all_results_correct": all(r["correct"] for r in runs if not r["deadlock"]),
            "median_seconds_when_ok": float(np.median(good)) if good else None,
            "trials": runs,
        }
        print(f"{key:20s}: deadlocked {dl}/{n_trials} trials")

    # How many collectives does it take before an unscheduled pair deadlocks?
    sweep = []
    for k in (1, 2, 4, 8, 16):
        runs = [trial("unscheduled_fixed", k, timeout) for _ in range(n_trials)]
        rate = sum(r["deadlock"] for r in runs) / n_trials
        sweep.append({"ops_per_client": k, "deadlock_rate": rate})
        print(f"unscheduled_fixed, {k:2d} ops/client: deadlock rate {rate:.2f}")
    out["sweep_unscheduled_fixed"] = sweep
    save("e2_gang", out)


if __name__ == "__main__":
    main()
