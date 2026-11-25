"""A small decoder-only transformer in plain JAX, built to be cut into pipeline stages.

Everything is a pure function of explicit pytrees, so the same code serves as
a single program (``n_stages=1``), as data-parallel replicas, and as stages of
an MPMD pipeline. Stage ``i`` gets its own forward and backward programs, and
the runtime wires them together with point-to-point transfers.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Config:
    vocab: int = 65
    ctx: int = 64
    d: int = 128
    heads: int = 4
    layers: int = 4
    batch: int = 32  # sequences per rank per step
    lr: float = 2e-3
    warmup: int = 50
    b1: float = 0.9
    b2: float = 0.99
    clip: float = 1.0


# ----------------------------------------------------------------- params


def _dense(key, fan_in, fan_out, scale=1.0):
    return jax.random.normal(key, (fan_in, fan_out), jnp.float32) * (scale / np.sqrt(fan_in))


def init_block(key, cfg: Config) -> dict:
    k = jax.random.split(key, 4)
    d = cfg.d
    resid = 1.0 / np.sqrt(2 * cfg.layers)
    return {
        "ln1": {"g": jnp.ones(d), "b": jnp.zeros(d)},
        "qkv": _dense(k[0], d, 3 * d),
        "proj": _dense(k[1], d, d, resid),
        "ln2": {"g": jnp.ones(d), "b": jnp.zeros(d)},
        "w1": _dense(k[2], d, 4 * d),
        "b1": jnp.zeros(4 * d),
        "w2": _dense(k[3], 4 * d, d, resid),
        "b2": jnp.zeros(d),
    }


def stage_layers(cfg: Config, n_stages: int) -> list[list[int]]:
    per = np.array_split(np.arange(cfg.layers), n_stages)
    return [list(map(int, p)) for p in per]


def init(seed: int, cfg: Config, n_stages: int = 1) -> list[dict]:
    """Return one param pytree per stage. Splitting never changes the init."""
    key = jax.random.PRNGKey(seed)
    k_emb, k_pos, k_head, k_blocks = jax.random.split(key, 4)
    blocks = [init_block(k, cfg) for k in jax.random.split(k_blocks, cfg.layers)]
    stages = []
    for i, layer_ids in enumerate(stage_layers(cfg, n_stages)):
        p = {"blocks": [blocks[j] for j in layer_ids]}
        if i == 0:
            p["emb"] = jax.random.normal(k_emb, (cfg.vocab, cfg.d)) * 0.02
            p["pos"] = jax.random.normal(k_pos, (cfg.ctx, cfg.d)) * 0.02
        if i == n_stages - 1:
            p["lnf"] = {"g": jnp.ones(cfg.d), "b": jnp.zeros(cfg.d)}
            p["head"] = _dense(k_head, cfg.d, cfg.vocab)
        stages.append(p)
    return stages


def n_params(tree) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree.leaves(tree)))


# ---------------------------------------------------------------- forward


def _ln(p, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + 1e-5) * p["g"] + p["b"]


def _block(p, x, heads):
    B, T, D = x.shape
    h = _ln(p["ln1"], x)
    q, k, v = jnp.split(h @ p["qkv"], 3, axis=-1)
    q, k, v = (t.reshape(B, T, heads, D // heads).transpose(0, 2, 1, 3) for t in (q, k, v))
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(D // heads)
    mask = jnp.tril(jnp.ones((T, T), bool))
    att = jax.nn.softmax(jnp.where(mask, att, -1e9), axis=-1)
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    x = x + y @ p["proj"]
    h = _ln(p["ln2"], x)
    return x + jax.nn.gelu(h @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def stage_apply(p, x, cfg: Config):
    """Stage forward. ``x`` is int tokens on the first stage, activations elsewhere."""
    if "emb" in p:
        x = p["emb"][x] + p["pos"][: x.shape[1]]
    for bp in p["blocks"]:
        x = _block(bp, x, cfg.heads)
    if "head" in p:
        x = _ln(p["lnf"], x) @ p["head"]
    return x


def xent(logits, targets):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None].astype(jnp.int32), axis=-1).mean()


# ------------------------------------------------------- program factories


def make_programs(cfg: Config, n_stages: int = 1, n_micro: int = 1) -> dict:
    """Pure functions the runtime compiles. Keys are program names.

    Pipeline stage ``i`` uses ``fwd_i``, ``bwd_i`` (``last`` for the final
    stage, which fuses forward, loss and backward), plus the shared ``add``
    and ``adam`` programs. ``n_micro`` scales the loss so that summed
    micro-batch gradients equal full-batch gradients.
    """
    scale = 1.0 / n_micro

    def fwd(p, x):
        return stage_apply(p, x, cfg)

    def bwd(p, x, gy):
        _, vjp = jax.vjp(lambda p_, x_: stage_apply(p_, x_, cfg), p, x)
        gp, gx = vjp(gy)
        return gp, gx

    def bwd_first(p, x, gy):  # tokens have no gradient
        _, vjp = jax.vjp(lambda p_: stage_apply(p_, x, cfg), p)
        return vjp(gy)[0]

    def last(p, x, y):
        def f(p_, x_):
            return xent(stage_apply(p_, x_, cfg), y) * scale

        loss, (gp, gx) = jax.value_and_grad(f, argnums=(0, 1))(p, x)
        return loss / scale, gp, gx

    def loss_and_grad(p, x, y):  # the whole model on one device
        return jax.value_and_grad(lambda p_: xent(stage_apply(p_, x, cfg), y) * scale)(p)

    def eval_loss(p, x, y):
        return xent(stage_apply(p, x, cfg), y)

    def add(a, b):
        return jax.tree.map(jnp.add, a, b)

    def sqnorm(g):
        return sum(jnp.sum(x * x) for x in jax.tree.leaves(g))

    def adam(p, m, v, g, step):
        return adam_global(p, m, v, g, step, sqnorm(g))

    def adam_global(p, m, v, g, step, sq):
        """Adam with clipping by a norm computed elsewhere (e.g. across pipeline stages)."""
        step = step + 1.0
        if cfg.clip:
            norm = jnp.sqrt(sq)
            g = jax.tree.map(lambda x: x * jnp.minimum(1.0, cfg.clip / (norm + 1e-6)), g)
        lr = cfg.lr * jnp.minimum(1.0, step / cfg.warmup)
        m = jax.tree.map(lambda m_, g_: cfg.b1 * m_ + (1 - cfg.b1) * g_, m, g)
        v = jax.tree.map(lambda v_, g_: cfg.b2 * v_ + (1 - cfg.b2) * g_ * g_, v, g)
        mh = 1 - cfg.b1**step
        vh = 1 - cfg.b2**step
        p = jax.tree.map(lambda p_, m_, v_: p_ - lr * (m_ / mh) / (jnp.sqrt(v_ / vh) + 1e-8), p, m, v)
        return p, m, v, step

    return {
        "fwd": fwd,
        "bwd": bwd,
        "bwd_first": bwd_first,
        "last": last,
        "loss_and_grad": loss_and_grad,
        "eval_loss": eval_loss,
        "add": add,
        "adam": adam,
        "adam_global": adam_global,
        "sqnorm": sqnorm,
    }


def make_batcher(cfg: Config, batch: int | None = None):
    """``batch(tokens, step, rank) -> (x, y)``, deterministic in (step, rank).

    Each rank samples its own shard on-device, so the controller never
    touches training data: it only sends two integers per step.
    """
    b = batch or cfg.batch

    def get_batch(tokens, step, rank):
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(1234), step), rank)
        starts = jax.random.randint(key, (b,), 0, tokens.shape[0] - cfg.ctx - 1)
        idx = starts[:, None] + jnp.arange(cfg.ctx + 1)[None, :]
        seq = tokens[idx].astype(jnp.int32)
        return seq[:, :-1], seq[:, 1:]

    return get_batch


def zeros_like(tree):
    return jax.tree.map(lambda x: np.zeros(x.shape, np.float32), tree)


def generate(
    params: dict, cfg: Config, vocab: list[str], prompt: str, n: int, temperature: float = 0.8, seed: int = 0
) -> str:
    """Sample text from a single-stage param tree."""
    lut = {ch: i for i, ch in enumerate(vocab)}
    ids = [lut[ch] for ch in prompt]
    fwd = jax.jit(lambda p, x: stage_apply(p, x, cfg)[:, -1])
    key = jax.random.PRNGKey(seed)
    for _ in range(n):
        ctx = np.array(ids[-cfg.ctx :], np.int32)[None]
        ctx = np.pad(ctx, ((0, 0), (cfg.ctx - ctx.shape[1], 0)))  # left-pad so the jit shape is fixed
        logits = fwd(params, ctx)[0] / temperature
        key, sub = jax.random.split(key)
        ids.append(int(jax.random.categorical(sub, logits)))
    return "".join(vocab[i] for i in ids)
