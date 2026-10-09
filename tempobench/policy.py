"""Planners that produce 50-row chunks: an oracle (the expert rolled forward) and a learned
flow-matching chunk policy (same recipe as the Green-VLA action expert: rectified-flow target,
10 Euler integration steps at inference, deltas to the observed state)."""
from __future__ import annotations

import functools
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import world as W

DSCALE = 0.25          # xy delta normalisation


# ------------------------------------------------------------------------------- oracle
def oracle_plan(world, delta_demo=W.lag_of(W.OMEGA_DEMO), tempo=1.0):
    """What the demonstration would do from here: the expert and a demo-robot copy rolled
    forward H frames, with the expert's command initialised the way it relates to the hand in
    the data (command = hand + delta_demo * velocity)."""
    n = world.n
    sh = W.World.__new__(W.World)
    sh.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in world.__dict__.items()})
    sh.omega = W.OMEGA_DEMO
    sh.timeouts = np.array([1e9, 1e9])
    sh.t_phase_start = np.full(n, sh.t)
    ex = W.Expert(sh, tempo)
    ex.c = world.p + delta_demo * world.v
    ex.cv = world.v.copy()
    ex.cg = np.where(world.holding | ((world.phase == 0) & (world.g > 0.5)), 1.0, 0.0)
    rows = np.zeros((n, W.H, 3))
    for j in range(W.H):
        c, cg = ex.step()
        rows[:, j, :2] = c - world.p
        rows[:, j, 2] = cg
        sh.step(c, cg, W.FRAME)
    return rows


# ------------------------------------------------------------------------------- flow policy
def _dense(key, din, dout, scale=1.0):
    return {"w": jax.random.normal(key, (din, dout)) * np.sqrt(2.0 / din) * scale, "b": jnp.zeros(dout)}


def init_params(key, obs_dim=10, width=512, depth=4):
    ks = jax.random.split(key, depth + 3)
    din = obs_dim + W.H * 3 + 32
    p = {"temb": _dense(ks[0], 1, 32), "layers": []}
    d = din
    for i in range(depth):
        p["layers"].append(_dense(ks[i + 1], d, width))
        d = width
    p["out"] = _dense(ks[-1], width, W.H * 3, 0.01)
    return p


def _norm_obs(o):
    return o * 2.0 - 1.0


def apply(p, obs, x, t):
    te = jnp.sin(t[:, None] * p["temb"]["w"] * 6.0 + p["temb"]["b"])
    h = jnp.concatenate([_norm_obs(obs), x, te], -1)
    for l in p["layers"]:
        h = jax.nn.gelu(h @ l["w"] + l["b"])
    return h @ p["out"]["w"] + p["out"]["b"]


def encode(rows):
    """(B,H,3) rows -> normalised flat (B,150): xy/DSCALE, gripper -> [-1,1]."""
    r = np.asarray(rows, np.float32).copy()
    r[..., :2] /= DSCALE
    r[..., 2] = r[..., 2] * 2.0 - 1.0
    return r.reshape(len(r), -1)


def decode(x):
    r = np.asarray(x, np.float64).reshape(len(x), W.H, 3).copy()
    r[..., :2] *= DSCALE
    r[..., 2] = np.clip((r[..., 2] + 1.0) / 2.0, 0, 1)
    return r


def train(X, Y, seed=0, steps=12000, bs=256, lr=1e-3, init=None, log_every=2000, tag=""):
    key = jax.random.PRNGKey(seed)
    params = init_params(key) if init is None else init
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, 300, steps, lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-5))
    st = opt.init(params)
    Yn = encode(Y)

    def loss_fn(p, o, x1, k):
        k1, k2 = jax.random.split(k)
        x0 = jax.random.normal(k1, x1.shape)
        t = jax.random.uniform(k2, (x1.shape[0],))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        return ((apply(p, o, xt, t) - (x1 - x0)) ** 2).mean()

    @jax.jit
    def upd(p, s, o, x1, k):
        l, g = jax.value_and_grad(loss_fn)(p, o, x1, k)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    rng = np.random.RandomState(seed)
    for it in range(steps):
        i = rng.randint(len(X), size=bs)
        key, k = jax.random.split(key)
        params, st, l = upd(params, st, X[i], Yn[i], k)
        if log_every and (it + 1) % log_every == 0:
            print(f"    [flow{tag}] {it+1}/{steps} loss {float(l):.4f}", flush=True)
    return params


@functools.partial(jax.jit, static_argnums=(3,))
def _sample(p, obs, x0, n_steps):
    x = x0
    for k in range(n_steps):
        t = jnp.full((obs.shape[0],), k / n_steps)
        x = x + apply(p, obs, x, t) / n_steps
    return x


class FlowPolicy:
    def __init__(self, params, seed=0, n_steps=10):
        self.p, self.n_steps = params, n_steps
        self.key = jax.random.PRNGKey(seed + 999)

    def sample(self, obs, noise=None):
        if noise is None:
            self.key, k = jax.random.split(self.key)
            noise = jax.random.normal(k, (obs.shape[0], W.H * 3))
        return decode(_sample(self.p, jnp.asarray(obs), noise, self.n_steps))

    def __call__(self, world):
        return self.sample(world.obs())


def save(params, path):
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)


def load(path):
    with open(path, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))


def open_loop_error(policy, X, Y, n=2000, seed=0, rows=slice(0, 25)):
    """Green-style offline metric: mean |sampled chunk - demonstrated chunk| (xy), first rows."""
    rng = np.random.RandomState(seed)
    i = rng.choice(len(X), min(n, len(X)), replace=False)
    pred = policy.sample(X[i])
    return float(np.abs(pred[:, rows, :2] - Y[i][:, rows, :2]).mean())


def planned_tempo(policy, X, Y=None, n=2000, seed=0, j0=2, j1=17):
    """Offline tempo proxy: speed of the plan's own motion over rows j0..j1 (demo time),
    restricted to cruising frames (demo speed > 60 % of max). Returns (policy, demo) medians."""
    rng = np.random.RandomState(seed)
    i = rng.choice(len(X), min(n, len(X)), replace=False)
    pred = policy.sample(X[i])
    sp = np.linalg.norm(pred[:, j1, :2] - pred[:, j0, :2], axis=1) / ((j1 - j0) * W.FRAME)
    if Y is None:
        return float(np.median(sp)), None
    sd = np.linalg.norm(Y[i][:, j1, :2] - Y[i][:, j0, :2], axis=1) / ((j1 - j0) * W.FRAME)
    m = sd > 0.6 * W.VMAX
    return float(np.median(sp[m])), float(np.median(sd[m]))
