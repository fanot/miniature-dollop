"""Closed-loop evaluation, failure taxonomy and causal probes."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import env as E
from . import models as M

OUTCOMES = ["success", "wrong_object", "miss", "collision", "timeout", "air_at_usual_place"]


@functools.lru_cache(maxsize=None)
def _policy_jit(kind):
    def f(p, img, ee, cid, sid, anchor):
        a, gl, _ = M.policy_apply(p, kind, img.astype(jnp.float32) / 255.0, ee, cid, sid, anchor)
        return a, jax.nn.sigmoid(gl)
    return jax.jit(f)


@jax.jit
def _ground_jit(gp, img, cid, sid):
    lg = M.ground_logits(gp, img.astype(jnp.float32) / 255.0, cid, sid)
    B = lg.shape[0]
    idx = jnp.argmax(lg.reshape(B, -1), -1)
    r, c = idx // 32, idx % 32
    rr, cc = jnp.meshgrid(jnp.arange(32), jnp.arange(32), indexing="ij")
    near = (jnp.abs(rr[None] - r[:, None, None]) <= 2) & (jnp.abs(cc[None] - c[:, None, None]) <= 2)
    lg = jnp.where(near, lg, -1e9)                     # local soft-argmax around the peak
    return M.logits_to_xy(lg)


def scenes_to_arrays(scenes):
    return dict(pos=np.stack([s.pos for s in scenes]), ids=np.stack([s.ids for s in scenes]),
                theta=np.stack([s.theta for s in scenes]), present=np.stack([s.present for s in scenes]),
                bg=np.stack([s.bg for s in scenes]), ee0=np.stack([s.ee0 for s in scenes]),
                tid=np.array([s.target_id for s in scenes]))


REGROUND_EVERY = 5   # the pointing model runs at 1/5 of the control rate (as a VLM would)


def rollout(kind, params, scenes, gparams=None, anchor_mode="pred", anchor_noise=0.0,
            override_xy=None, guide_lambda=0.0, seed=0, batch=250):
    """Returns dict with per-episode outcome codes, grabbed slot, grounding hit and trajectories."""
    res = [_rollout_batch(kind, params, scenes[i:i + batch], gparams, anchor_mode, anchor_noise,
                          None if override_xy is None else override_xy[i:i + batch], guide_lambda, seed + i)
           for i in range(0, len(scenes), batch)]
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]}


def _rollout_batch(kind, params, scenes, gparams, anchor_mode, anchor_noise, override_xy, guide_lambda, seed):
    A = scenes_to_arrays(scenes)
    N = len(scenes)
    rng = np.random.RandomState(seed)
    cid = np.array([E.IDENTS[t][0] for t in A["tid"]])
    sid = np.array([E.IDENTS[t][1] for t in A["tid"]])
    ee = A["ee0"].copy()
    active = np.ones(N, bool)
    outcome = np.full(N, 4)                  # timeout by default
    grabbed = np.full(N, -1)
    traj = np.zeros((N, E.TMAX + 1, 2), np.float32)
    traj[:, 0] = ee
    anchors0 = np.zeros((N, 2), np.float32)
    offset = rng.normal(0, anchor_noise, (N, 2)).astype(np.float32) if anchor_noise > 0 else np.zeros((N, 2), np.float32)
    need_anchor = M.uses_anchor(kind) or guide_lambda > 0
    pol = _policy_jit(kind)
    u = E.unit(A["theta"])                                   # (N,K,2)
    gpts = A["pos"] + E.GRASP_OFF * u
    for t in range(E.TMAX):
        img = E.render_batch(A["pos"], A["ids"], A["theta"], A["present"], ee, A["bg"])
        if not need_anchor:
            anchor = np.zeros((N, 2), np.float32)
        elif anchor_mode == "pred":
            if t % REGROUND_EVERY == 0:
                pred = np.asarray(_ground_jit(gparams, img, cid, sid))
            anchor = pred.copy()
        elif anchor_mode == "oracle":
            anchor = A["pos"][:, 0].copy()
        elif anchor_mode == "override":
            anchor = override_xy.astype(np.float32).copy()
        else:
            raise ValueError(anchor_mode)
        anchor = np.clip(anchor + offset, 0, 1).astype(np.float32)
        if t == 0:
            anchors0 = anchor.copy()
        a, pg = pol(params, img, ee.astype(np.float32), cid, sid, anchor)
        a, pg = np.asarray(a), np.asarray(pg)
        if guide_lambda > 0:
            d = anchor - ee
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            far = dist > 0.25
            a = np.where(far, np.clip((1 - guide_lambda) * a + guide_lambda * d / np.maximum(dist, 1e-6), -1, 1), a)
        grip = active & (pg[:, 0] > 0.5 if pg.ndim == 2 else pg > 0.5)
        # --- evaluate grasps
        for n in np.where(grip)[0]:
            outcome[n], grabbed[n] = _judge(ee[n], A, gpts, n)
            active[n] = False
        # --- move the rest
        mv = active
        ee[mv] = np.clip(ee[mv] + a[mv] * E.STEP, 0, 1)
        dist_obj = np.linalg.norm(A["pos"] - ee[:, None], axis=-1)
        dist_obj[~A["present"]] = 9
        col = mv & (dist_obj.min(-1) < E.COLLIDE)
        outcome[col] = 3
        active[col] = False
        traj[:, t + 1] = ee
        if not active.any():
            break
    tgt = A["pos"][:, 0]
    ground_hit = np.linalg.norm(anchors0 - tgt, axis=-1) < 0.05
    return dict(outcome=outcome, grabbed=grabbed, ground_hit=ground_hit, traj=traj, anchor0=anchors0)


def _judge(ee, A, gpts, n):
    present = A["present"][n]
    if present[0] and np.linalg.norm(ee - gpts[n, 0]) < E.GRIP_TOL:
        return 0, 0
    for k in range(1, len(present)):
        if present[k] and (np.linalg.norm(ee - gpts[n, k]) < 0.045 or
                           np.linalg.norm(ee - A["pos"][n, k]) < E.R + 0.05):
            return 1, k
    if not present[0] and np.linalg.norm(ee - E.ANCHORS[A["tid"][n]]) < 0.15:
        return 5, -1
    return 2, -1


def summarize(r):
    out = {name: float(np.mean(r["outcome"] == i)) for i, name in enumerate(OUTCOMES)}
    out["grounding_acc"] = float(np.mean(r["ground_hit"]))
    out["n"] = int(len(r["outcome"]))
    return out
