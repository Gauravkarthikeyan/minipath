"""CartPole (Barto, Sutton & Anderson 1983) in JAX, plus a REINFORCE learner.

The dynamics and constants follow the classic control formulation used by
Gymnasium's ``CartPole-v1``. Episodes are capped at ``horizon`` steps. The
sampler and the learner are separate programs, so the runtime can place them
on separate slices, the same split as RL post-training (generation vs. training).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
THETA_LIMIT, X_LIMIT = 12 * 2 * np.pi / 360, 2.4


@dataclass(frozen=True)
class RLConfig:
    envs: int = 256  # parallel environments per sampler worker
    horizon: int = 200
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    minibatch_frac: int = 4  # learner trains on 1/minibatch_frac of the samples


def step_env(s, a):
    x, x_dot, th, th_dot = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    force = jnp.where(a == 1, FORCE, -FORCE)
    cos, sin = jnp.cos(th), jnp.sin(th)
    total = M_CART + M_POLE
    pml = M_POLE * HALF_LEN
    temp = (force + pml * th_dot**2 * sin) / total
    th_acc = (GRAVITY * sin - cos * temp) / (HALF_LEN * (4.0 / 3.0 - M_POLE * cos**2 / total))
    x_acc = temp - pml * th_acc * cos / total
    x, x_dot = x + TAU * x_dot, x_dot + TAU * x_acc
    th, th_dot = th + TAU * th_dot, th_dot + TAU * th_acc
    s = jnp.stack([x, x_dot, th, th_dot], -1)
    done = (jnp.abs(x) > X_LIMIT) | (jnp.abs(th) > THETA_LIMIT)
    return s, done


def init_policy(seed: int, cfg: RLConfig) -> dict:
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    h = cfg.hidden
    return {
        "w1": jax.random.normal(k1, (4, h)) * 0.5,
        "b1": jnp.zeros(h),
        "w2": jax.random.normal(k2, (h, h)) / np.sqrt(h),
        "b2": jnp.zeros(h),
        "w3": jax.random.normal(k3, (h, 2)) * 0.01,
        "b3": jnp.zeros(2),
    }


def logits_fn(p, obs):
    h = jnp.tanh(obs @ p["w1"] + p["b1"])
    h = jnp.tanh(h @ p["w2"] + p["b2"])
    return h @ p["w3"] + p["b3"]


def make_programs(cfg: RLConfig) -> dict:
    def rollout(p, step, rank):
        """Play ``envs`` episodes with the given policy. Returns a training batch."""
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(7), step), rank)
        k0, key = jax.random.split(key)
        s0 = jax.random.uniform(k0, (cfg.envs, 4), minval=-0.05, maxval=0.05)

        def body(carry, k):
            s, alive = carry
            a = jax.random.categorical(k, logits_fn(p, s))
            s2, done = step_env(s, a)
            r = alive.astype(jnp.float32)
            return (s2, alive & ~done), (s, a, r)

        keys = jax.random.split(key, cfg.horizon)
        _, (obs, act, rew) = jax.lax.scan(body, (s0, jnp.ones(cfg.envs, bool)), keys)

        def disc(g, r):
            g = r + cfg.gamma * g
            return g, g

        _, ret = jax.lax.scan(disc, jnp.zeros(cfg.envs), rew, reverse=True)
        mask = rew  # 1 while the episode is alive
        ep_return = rew.sum(0).mean()
        return obs, act, ret, mask, ep_return

    def loss_and_grad(p, obs, act, ret, mask, step):
        # Per-timestep baseline: average return-to-go across envs at each t.
        # Once every episode survives to the cap, advantages go to zero instead
        # of becoming time-index noise that Adam would amplify.
        alive = mask.sum(1, keepdims=True)
        base = (ret * mask).sum(1, keepdims=True) / jnp.maximum(alive, 1.0)
        n = mask.sum()
        dev = (ret - base) * mask
        adv = dev / (jnp.sqrt((dev**2).sum() / n) + 1e-6)

        # Train on a random quarter of the (t, env) samples. Generation then
        # dominates the iteration, as it does in RL post-training of LLMs.
        T, B = mask.shape
        k = (T * B) // cfg.minibatch_frac
        idx = jax.random.choice(jax.random.PRNGKey(step), T * B, (k,), replace=False)
        o, a, ad, mk = obs.reshape(T * B, 4)[idx], act.reshape(-1)[idx], adv.reshape(-1)[idx], mask.reshape(-1)[idx]

        def loss(p_):
            logp = jax.nn.log_softmax(logits_fn(p_, o))
            lp = jnp.take_along_axis(logp, a[:, None], -1)[:, 0]
            return -(lp * ad * mk).sum() / jnp.maximum(mk.sum(), 1.0)

        return jax.value_and_grad(loss)(p)

    def adam(p, m, v, g, t):
        t = t + 1.0
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree.map(
            lambda p_, m_, v_: p_ - cfg.lr * (m_ / (1 - 0.9**t)) / (jnp.sqrt(v_ / (1 - 0.999**t)) + 1e-8), p, m, v
        )
        return p, m, v, t

    return {"rollout": rollout, "loss_and_grad": loss_and_grad, "adam": adam}
