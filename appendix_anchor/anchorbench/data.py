"""Dataset generation: robot demonstrations (narrow) + web-style pointing data (broad)."""
from __future__ import annotations

import numpy as np

from . import env as E


def generate_demos(n_episodes: int, seed: int, noise_std: float = 0.35, p_noisy: float = 0.5,
                   anchor_sigma: float = 0.04, p_uniform: float = 0.0):
    """Scripted expert with DART-style noise injection (executed action is perturbed,
    the label is always the clean expert action at the visited state)."""
    rng = np.random.RandomState(seed)
    rec = {k: [] for k in ["pos", "ids", "theta", "present", "ee", "bg", "act", "grip",
                           "cid", "sid", "tgt_xy", "ep"]}
    for ep in range(n_episodes):
        sc = E.sample_scene(rng, "id", anchor_sigma=anchor_sigma, p_uniform=p_uniform)
        noisy = rng.rand() < p_noisy
        ee = sc.ee0.copy()
        c, th = sc.pos[0], sc.theta[0]
        cc, ss = E.IDENTS[sc.target_id]
        for t in range(70):
            a, grip = E.expert_action(ee, c, th)
            rec["pos"].append(sc.pos); rec["ids"].append(sc.ids); rec["theta"].append(sc.theta)
            rec["present"].append(sc.present); rec["bg"].append(sc.bg); rec["ee"].append(ee.copy())
            rec["act"].append(a); rec["grip"].append(float(grip)); rec["cid"].append(cc)
            rec["sid"].append(ss); rec["tgt_xy"].append(c); rec["ep"].append(ep)
            if grip:
                break
            a_exec = a
            if noisy:
                a_n = np.clip(a + rng.normal(0, noise_std, 2), -1, 1).astype(np.float32)
                ee_n = np.clip(ee + a_n * E.STEP, 0, 1)
                if not E._collides(ee_n, sc):
                    a_exec = a_n
            ee = np.clip(ee + a_exec * E.STEP, 0, 1)
            if E._collides(ee, sc):
                break
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["img"] = render_records(out)
    return out


def generate_pointing(n: int, seed: int):
    rng = np.random.RandomState(seed + 10_000)
    rec = {k: [] for k in ["pos", "ids", "theta", "present", "ee", "bg", "cid", "sid", "tgt_xy"]}
    K = 4
    for _ in range(n):
        sc, k = E.sample_pointing_scene(rng)
        sc.bg = rng.uniform(0.05, 0.5, 3).astype(np.float32)   # web images are visually diverse
        # pad to K slots with absent objects so that rendering can be batched
        pad = K - sc.K
        rec["pos"].append(np.concatenate([sc.pos, np.zeros((pad, 2), np.float32)]))
        rec["ids"].append(np.concatenate([sc.ids, np.zeros(pad, np.int64)]))
        rec["theta"].append(np.concatenate([sc.theta, np.zeros(pad, np.float32)]))
        rec["present"].append(np.concatenate([sc.present, np.zeros(pad, bool)]))
        rec["ee"].append(sc.ee0); rec["bg"].append(sc.bg)
        cc, ss = E.IDENTS[sc.ids[k]]
        rec["cid"].append(cc); rec["sid"].append(ss); rec["tgt_xy"].append(sc.pos[k])
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["img"] = render_records(out)
    return out


def render_records(r, chunk=4096):
    imgs = []
    for i in range(0, len(r["ids"]), chunk):
        s = slice(i, i + chunk)
        imgs.append(E.render_batch(r["pos"][s], r["ids"][s], r["theta"][s], r["present"][s],
                                   r["ee"][s], r["bg"][s]))
    return np.concatenate(imgs)
