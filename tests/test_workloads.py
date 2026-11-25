"""Workload correctness: pipeline == single device, recovery is exact, CartPole physics."""

from __future__ import annotations

import numpy as np

from experiments import e3_pipeline as E3
from experiments import e4_fault_tolerance as E4
from minipath import Cluster
from minipath.workloads import cartpole as CP
from minipath.workloads import transformer as T

TINY = T.Config(d=32, heads=2, layers=4, ctx=16, batch=8)


def fake_tokens(n=20_000, seed=0):
    return np.random.default_rng(seed).integers(0, TINY.vocab, n).astype(np.uint8)


def test_pipeline_matches_single_device():
    toks = fake_tokens()
    with Cluster(4) as c:
        pipe = E3.Pipeline(c, TINY, 2, toks)
        losses = [pipe.step(k) for k in range(4)]
        pipe_curve = [float(np.mean([float(f.fetch(0)) for f in ls])) for ls in losses]
        for s in pipe.stages:
            c.rm.release(s)
        single_curve, _ = E3.single_device_run(c, TINY, 2, toks, 4)
    np.testing.assert_allclose(pipe_curve, single_curve, atol=1e-5)
    assert pipe_curve[-1] < pipe_curve[0]


def test_recovery_is_bit_identical_to_clean_run():
    toks = fake_tokens()
    total = 12
    with Cluster(3) as c:
        clean = E4.DPTrainer(c, TINY, toks, n=3).train(total)
    with Cluster(3, n_spares=1) as c:
        tr = E4.DPTrainer(c, TINY, toks, n=3)
        log = tr.train(total, kill_at={5: 1})
        assert any(e["kind"] == "remap" for e in tr.events)
    a = {r["step"]: r["loss"] for r in clean}
    b = {r["step"]: r["loss"] for r in log}
    assert set(a) == set(b) == set(range(total))
    assert all(a[k] == b[k] for k in a), "recovered losses differ from the clean run"


def test_elastic_shrink_keeps_training():
    toks = fake_tokens()
    with Cluster(3) as c:
        tr = E4.DPTrainer(c, TINY, toks, n=3)
        log = tr.train(10, kill_at={4: 2})
    assert any(e["kind"] == "shrink" for e in tr.events)
    assert max(r["step"] for r in log) == 9
    assert log[-1]["replicas"] == 2


def test_cartpole_physics_matches_reference_step():
    # One step from a known state, checked against the closed-form equations
    s = np.array([[0.01, 0.02, 0.03, 0.04]], np.float32)
    s2, done = CP.step_env(s, np.array([1]))
    x, xd, th, thd = s[0]
    total, pml = CP.M_CART + CP.M_POLE, CP.M_POLE * CP.HALF_LEN
    temp = (CP.FORCE + pml * thd**2 * np.sin(th)) / total
    tha = (CP.GRAVITY * np.sin(th) - np.cos(th) * temp) / (CP.HALF_LEN * (4 / 3 - CP.M_POLE * np.cos(th) ** 2 / total))
    xa = temp - pml * tha * np.cos(th) / total
    want = [x + CP.TAU * xd, xd + CP.TAU * xa, th + CP.TAU * thd, thd + CP.TAU * tha]
    np.testing.assert_allclose(np.asarray(s2)[0], want, rtol=1e-5)
    assert not bool(done[0])
