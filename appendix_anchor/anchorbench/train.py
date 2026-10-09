"""Training loops (behaviour cloning + pointing)."""
from __future__ import annotations

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import models as M

ANCHOR_TRAIN_NOISE = 0.015   # grounded-point noise used when training anchor-conditioned heads


def _opt(steps, lr=1e-3):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps, lr * 0.01)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))


def _gauss_target(xy, n):
    """Soft one-hot over an n x n grid around the (x,y) target (sigma = 1 cell)."""
    g = (jnp.arange(n) + 0.5) / n
    cx = xy[:, 0][:, None]
    cy = (1.0 - xy[:, 1])[:, None]
    wx = jnp.exp(-0.5 * ((g[None] - cx) * n) ** 2)
    wy = jnp.exp(-0.5 * ((g[None] - cy) * n) ** 2)
    t = wy[:, :, None] * wx[:, None, :]
    return t / t.sum((1, 2), keepdims=True)


def _point_ce(logits, xy):
    B, n, _ = logits.shape
    t = _gauss_target(xy, n).reshape(B, -1)
    return -(t * jax.nn.log_softmax(logits.reshape(B, -1), -1)).sum(-1).mean()


def _f(img):
    return img.astype(jnp.float32) / 255.0


# ------------------------------------------------------------------ grounding
def train_grounding(robot, web, seed, steps=2500, bs=64, log_every=500):
    key = jax.random.PRNGKey(seed)
    params = M.init_ground(key)
    opt = _opt(steps)
    st = opt.init(params)

    def loss_fn(p, img, cid, sid, xy):
        lg = M.ground_logits(p, _f(img), cid, sid)
        ce = _point_ce(lg, xy)
        l2 = ((M.logits_to_xy(lg) - xy) ** 2).sum(-1).mean()
        return ce + 20.0 * l2, (ce, l2)

    @jax.jit
    def update(p, s, img, cid, sid, xy):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, img, cid, sid, xy)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l, aux

    rng = np.random.RandomState(seed)
    t0 = time.time()
    for it in range(steps):
        ir = rng.randint(len(robot["img"]), size=bs // 2)
        iw = rng.randint(len(web["img"]), size=bs // 2)
        img = np.concatenate([robot["img"][ir], web["img"][iw]])
        cid = np.concatenate([robot["cid"][ir], web["cid"][iw]])
        sid = np.concatenate([robot["sid"][ir], web["sid"][iw]])
        xy = np.concatenate([robot["tgt_xy"][ir], web["tgt_xy"][iw]]).astype(np.float32)
        params, st, l, (ce, l2) = update(params, st, img, cid, sid, xy)
        if (it + 1) % log_every == 0:
            print(f"  [ground] it {it+1}/{steps} loss {float(l):.3f} ce {float(ce):.3f} "
                  f"rmse {float(jnp.sqrt(l2)):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return params


# ------------------------------------------------------------------ policies
def train_policy(kind, robot, web, seed, steps=4000, bs=128, log_every=1000):
    key = jax.random.PRNGKey(seed * 100 + M.KINDS.index(kind))
    params = M.init_policy(key, kind)
    opt = _opt(steps)
    st = opt.init(params)
    cotrain = kind == "bc_cotrain"

    def loss_fn(p, img, ee, cid, sid, anchor, act, grip, tgt, wimg, wee, wcid, wsid, wtgt):
        a, gl, aux = M.policy_apply(p, kind, _f(img), ee, cid, sid, anchor)
        l_act = ((a - act) ** 2).sum(-1).mean()
        l_grip = optax.sigmoid_binary_cross_entropy(gl, grip)
        l_grip = (l_grip * jnp.where(grip > 0.5, 4.0, 1.0)).mean()
        loss = l_act + 0.5 * l_grip
        if cotrain:
            _, _, waux = M.policy_apply(p, kind, _f(wimg), wee, wcid, wsid, wtgt)
            loss = loss + 0.1 * (_point_ce(aux, tgt) + _point_ce(waux, wtgt))
        return loss, (l_act, l_grip)

    @jax.jit
    def update(p, s, *batch):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, *batch)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l, aux

    rng = np.random.RandomState(seed + 7)
    t0 = time.time()
    n = len(robot["img"])
    for it in range(steps):
        i = rng.randint(n, size=bs)
        tgt = robot["tgt_xy"][i].astype(np.float32)
        anchor = (tgt + rng.normal(0, ANCHOR_TRAIN_NOISE, tgt.shape)).astype(np.float32)
        if cotrain:
            j = rng.randint(len(web["img"]), size=bs // 2)
            w = (web["img"][j], web["ee"][j], web["cid"][j], web["sid"][j], web["tgt_xy"][j].astype(np.float32))
        else:
            w = (robot["img"][i[:2]], robot["ee"][i[:2]], robot["cid"][i[:2]], robot["sid"][i[:2]], tgt[:2])
        params, st, l, (la, lg) = update(params, st, robot["img"][i], robot["ee"][i], robot["cid"][i],
                                         robot["sid"][i], anchor, robot["act"][i],
                                         robot["grip"][i].astype(np.float32), tgt, *w)
        if (it + 1) % log_every == 0:
            print(f"  [{kind}] it {it+1}/{steps} loss {float(l):.4f} act {float(la):.4f} "
                  f"grip {float(lg):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return params


def save(params, path):
    with open(path, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)


def load(path):
    with open(path, "rb") as f:
        return jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
