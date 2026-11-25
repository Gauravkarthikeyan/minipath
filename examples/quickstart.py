"""Data-parallel training of a char transformer on Tiny Shakespeare, driven by one controller.

    python examples/quickstart.py

Four workers, one Python loop. The loop never waits on compute. It only
blocks when it prints a loss.
"""

import numpy as np

from minipath import Cluster
from minipath.workloads import shakespeare
from minipath.workloads import transformer as T


def main():
    cfg = T.Config()
    tokens, _, vocab = shakespeare.load()
    progs = T.make_programs(cfg)
    params = T.init(seed=0, cfg=cfg)[0]
    zeros = T.zeros_like(params)

    with Cluster(4) as cluster:
        dp = cluster.allocate(4)  # 4 virtual devices
        batch = cluster.compile(T.make_batcher(cfg), n_out=2)  # plain JAX functions
        lg = cluster.compile(progs["loss_and_grad"], n_out=2)
        adam = cluster.compile(progs["adam"], n_out=4)

        data = cluster.put(tokens, dp)  # replicated once; batches are sampled on-device
        p, m, v, t = (cluster.put(x, dp) for x in (params, zeros, zeros, np.float32(0)))

        for step in range(200):
            ranks = [(np.int32(step), np.int32(r)) for r in range(4)]
            x, y = cluster.run(batch, dp, data, consts=ranks)
            loss, g = cluster.run(lg, dp, p, x, y)
            g = cluster.allreduce(g)  # ring all-reduce, gang-scheduled
            p, m, v, t = cluster.run(adam, dp, p, m, v, g, t)
            if step % 25 == 0:
                print(f"step {step:3d}  loss {float(loss.fetch(0)):.3f}")  # the only blocking call

        print(T.generate(p.fetch(0), cfg, vocab, "ROMEO:", 200))


if __name__ == "__main__":
    main()
