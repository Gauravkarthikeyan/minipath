"""Runtime semantics: futures, collectives, transfers, errors, scheduling, failures."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from minipath import Cluster, RemoteOpError, Sharded, WorkerLost


@pytest.fixture(scope="module")
def cluster():
    with Cluster(4) as c:
        yield c


def add(a, b):
    return a + b


def scale(x, k):
    return x * k


def slow_identity(x):
    time.sleep(0.5)
    return x


def boom(x):
    raise ValueError("intentional failure")


def test_put_run_fetch(cluster):
    s = cluster.allocate(2)
    try:
        x = cluster.put([np.full(3, 1.0, np.float32), np.full(3, 2.0, np.float32)], s)
        y = cluster.run(cluster.compile(add), s, x, x)
        np.testing.assert_array_equal(y.fetch(0), [2, 2, 2])
        np.testing.assert_array_equal(y.fetch(1), [4, 4, 4])
    finally:
        cluster.rm.release(s)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_ring_allreduce_matches_numpy(cluster, n, reduce):
    s = cluster.allocate(n)
    try:
        rng = np.random.default_rng(n)
        vals = [
            {"a": rng.standard_normal((5, 7)).astype(np.float32), "b": rng.standard_normal(11).astype(np.float32)}
            for _ in range(n)
        ]
        out = cluster.allreduce(cluster.put(vals, s), reduce=reduce).fetch()
        for key in ("a", "b"):
            want = np.sum([v[key] for v in vals], axis=0)
            if reduce == "mean":
                want = want / n
            for o in out:
                np.testing.assert_allclose(o[key], want, rtol=1e-5, atol=1e-6)
        # every rank gets bit-identical results (recovery relies on this)
        for o in out[1:]:
            np.testing.assert_array_equal(o["a"], out[0]["a"])
    finally:
        cluster.rm.release(s)


def test_transfer_broadcast_and_one_to_one(cluster):
    a, b = cluster.allocate(1), cluster.allocate(3)
    try:
        x = cluster.put(np.arange(4, dtype=np.float32), a)
        y = cluster.transfer(x, b)
        for v in y.fetch():
            np.testing.assert_array_equal(v, np.arange(4))
        z = cluster.run(cluster.compile(scale), b, y, consts=[(np.float32(r),) for r in range(3)])
        back = cluster.transfer(z[2], a)
        np.testing.assert_array_equal(back.fetch(0), 2 * np.arange(4))
    finally:
        cluster.rm.release(a)
        cluster.rm.release(b)


def test_dispatch_is_asynchronous(cluster):
    s = cluster.allocate(1)
    try:
        prog = cluster.compile(slow_identity, jit=False)
        x = cluster.put(np.ones(2, np.float32), s)
        t0 = time.perf_counter()
        y = x
        for _ in range(3):
            y = cluster.run(prog, s, y)
        dispatch = time.perf_counter() - t0
        y.fetch(0)
        total = time.perf_counter() - t0
        assert dispatch < 0.2, f"dispatch blocked for {dispatch:.2f}s"
        assert total >= 1.4
    finally:
        cluster.rm.release(s)


def test_errors_propagate_through_dataflow(cluster):
    s = cluster.allocate(2)
    try:
        x = cluster.put(np.ones(2, np.float32), s)
        bad = cluster.run(cluster.compile(boom), s, x)
        downstream = cluster.run(cluster.compile(add), s, bad, x)
        with pytest.raises(RemoteOpError, match="intentional failure"):
            downstream.fetch(1)
        # the stream keeps working afterwards
        ok = cluster.run(cluster.compile(add), s, x, x)
        np.testing.assert_array_equal(ok.fetch(0), [2, 2])
    finally:
        cluster.rm.release(s)


def test_buffers_are_freed_when_futures_die(cluster):
    s = cluster.allocate(1)
    try:
        w = cluster.rm.physical(s.vdevs[0])
        cluster.barrier(s)
        before = cluster._request(w, lambda req: ("stats", req))["bytes"]
        x = cluster.put(np.zeros(1_000_000, np.float32), s)
        cluster.barrier(s)
        during = cluster._request(w, lambda req: ("stats", req))["bytes"]
        del x
        cluster.put(np.zeros(1, np.float32), s)  # next dispatch flushes the deletes
        cluster.barrier(s)
        after = cluster._request(w, lambda req: ("stats", req))["bytes"]
        assert during - before >= 4_000_000
        assert after - before < 1000
    finally:
        cluster.rm.release(s)


def test_spmd_mode(cluster):
    s = cluster.allocate(4)
    try:

        def prog(ctx, base):
            return ctx.allreduce(np.array([base + ctx.rank], np.float32), reduce="sum")

        out = cluster.spmd(prog, s, (10.0,))
        assert [float(o[0]) for o in out] == [46.0] * 4
    finally:
        cluster.rm.release(s)


def test_gang_scheduling_prevents_deadlock():
    with Cluster(3, gang_scheduling=True) as c:
        c.configure(collective_timeout=3.0)
        s = c.allocate(3)
        prog = c.compile(add)
        results = {}

        def client(name):
            x = c.put(np.ones(8, np.float32), s)
            for _ in range(20):
                x = c.allreduce(c.run(prog, s, x, x), reduce="mean")
            results[name] = float(x.fetch(0)[0])

        threads = [threading.Thread(target=client, args=(n,)) for n in "AB"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == {"A": 2.0**20, "B": 2.0**20}


def test_failure_detection_remap_and_continue():
    with Cluster(2, n_spares=1) as c:
        s = c.allocate(2)
        prog = c.compile(add)
        x = c.put(np.ones(4, np.float32), s)
        y = c.allreduce(c.run(prog, s, x, x), reduce="sum")
        np.testing.assert_array_equal(y.fetch(0), [4] * 4)
        survivor_val = y.buffers[0]

        c.kill(c.rm.physical(s.vdevs[1]))
        deadline = time.time() + 5
        while not c._needs_recovery and time.time() < deadline:
            time.sleep(0.01)
        with pytest.raises(WorkerLost):
            c.run(prog, s, x, x)

        lost = c.begin_recovery()
        assert [v for _, vs in lost for v in vs] == [s.vdevs[1]]
        new_w = c.rm.remap(s.vdevs[1])
        assert new_w == 2
        # rebuild the lost shard from the survivor and keep going on the same virtual slice
        restored = Sharded(s, [survivor_val, c.copy(survivor_val, new_w)])
        z = c.allreduce(c.run(prog, s, restored, restored), reduce="sum")
        np.testing.assert_array_equal(z.fetch(1), [16] * 4)
