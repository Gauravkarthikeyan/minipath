"""E1: what does a single controller cost, and when does it stop mattering?

This mirrors the micro-benchmark in the Pathways paper (Barham et al. 2022,
section 5.1). A chain of trivial gang-scheduled computations, each one a
program followed by an all-reduce, where every computation consumes the
previous one's output. Three ways to drive it:

* spmd:  multi-controller. Every worker runs the loop itself, with no controller.
* sync:  single controller that waits for each op before sending the next
         (an RPC-per-step design, like calling ``ray.get`` in a loop).
* async: single controller that dispatches ahead on futures (Pathways' parallel
         asynchronous dispatch).

Part A uses scalar compute and varies the worker count. Part B fixes 4
workers and grows the per-op compute to find where async catches up with SPMD.
Part B uses 3 workers on purpose. The controller needs a core of its own
(on Pathways it runs on a separate CPU host), and this machine has 4.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.common import save
from minipath import Cluster


def add_one(x):
    return x + 1.0


def matmul_step(x, w, s):
    import jax.numpy as jnp

    y = jnp.tanh(x @ w + s)
    return y, jnp.mean(y, keepdims=True)


def spmd_chain(ctx, n_iter, n):
    import jax
    import jax.numpy as jnp
    import numpy as np_

    if n == 0:
        f = jax.jit(lambda x: x + 1.0)
        x = np_.zeros((1,), np_.float32)
        f(x).block_until_ready()
        ctx.allreduce(x)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            x = ctx.allreduce(f(x))
        return time.perf_counter() - t0
    f = jax.jit(matmul_step)
    rng = np_.random.default_rng(ctx.rank)
    x = rng.standard_normal((n, n)).astype(np_.float32)
    w = (rng.standard_normal((n, n)) / np_.sqrt(n)).astype(np_.float32)
    s = np_.zeros((1,), np_.float32)
    y, st = f(x, w, s)
    jax.block_until_ready(y)
    ctx.allreduce(st)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x, st = f(x, w, s)
        s = ctx.allreduce(st)
    jnp.asarray(x).block_until_ready()
    return time.perf_counter() - t0


def controller_chain(c: Cluster, s, mode: str, n_iter: int, n: int) -> float:
    wait = mode == "sync"
    if n == 0:
        prog = c.compile(add_one)
        x = c.put(np.zeros((1,), np.float32), s)
        x = c.allreduce(c.run(prog, s, x))
        x.fetch(0)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            x = c.allreduce(c.run(prog, s, x, wait=wait), wait=wait)
        x.fetch()
        return time.perf_counter() - t0
    prog = c.compile(matmul_step, n_out=2)
    rng = np.random.default_rng(0)
    x = c.put([rng.standard_normal((n, n)).astype(np.float32) for _ in s], s)
    w = c.put([(rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32) for _ in s], s)
    st = c.put(np.zeros((1,), np.float32), s)
    x, st = c.run(prog, s, x, w, st)
    st = c.allreduce(st)
    st.fetch(0)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x, st = c.run(prog, s, x, w, st, wait=wait)
        st = c.allreduce(st, wait=wait)
    st.fetch()
    return time.perf_counter() - t0


def measure(n_workers: int, n: int, n_iter: int, repeats: int) -> dict:
    out = {}
    with Cluster(n_workers) as c:
        s = c.allocate(n_workers)
        for mode in ("spmd", "sync", "async"):
            times = []
            for _ in range(repeats):
                if mode == "spmd":
                    times.append(max(c.spmd(spmd_chain, s, (n_iter, n))))
                else:
                    times.append(controller_chain(c, s, mode, n_iter, n))
            med = float(np.median(times))
            out[mode] = {"sec_per_step": med / n_iter, "steps_per_sec": n_iter / med, "all": times}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    repeats = 2 if args.quick else 5

    part_a = []
    for nw in (1, 2, 4):
        r = measure(nw, 0, 300 if args.quick else 1000, repeats)
        part_a.append({"workers": nw, **r})
        print(
            f"[A] workers={nw}: " + "  ".join(f"{m}={r[m]['steps_per_sec']:.0f}/s" for m in ("spmd", "sync", "async"))
        )

    part_b = []
    sizes = (32, 128, 512) if args.quick else (16, 32, 64, 128, 256, 384, 512, 768, 1024)
    for n in sizes:
        flops = 2 * n**3
        n_iter = int(np.clip(4e9 / flops, 20, 500))
        if args.quick:
            n_iter = max(10, n_iter // 4)
        r = measure(3, n, n_iter, repeats)
        part_b.append({"n": n, "flops": flops, "iters": n_iter, **r})
        eff = r["spmd"]["sec_per_step"] / r["async"]["sec_per_step"]
        print(
            f"[B] n={n}: "
            + "  ".join(f"{m}={r[m]['sec_per_step'] * 1e3:.2f}ms" for m in ("spmd", "sync", "async"))
            + f"  async/spmd eff={eff:.2f}"
        )

    save("e1_dispatch", {"part_a": part_a, "part_b": part_b})


if __name__ == "__main__":
    main()
