"""Policies and the grounding (pointing) network, written in plain JAX.

Method zoo (all trained by behaviour cloning on the same demonstrations):
  bc           - end-to-end FiLM-CNN policy (RT-1-style language conditioning) on the global image,
                 plus an end-effector-centred crop (a "wrist camera") for precise final approach
  bc_cotrain   - bc + auxiliary pointing head co-trained on robot frames and web pointing data
                 (stand-in for VLM co-training / Green-VLA L1 stage)
  inject       - bc + grounded target point injected into the action head as an extra input
                 (stand-in for "direct action-head injection of a grounded point")
  guide        - bc_cotrain at train time; at test time actions are steered toward the grounded
                 point in the far field (stand-in for Green-VLA JPM guidance; see evaluate.py)
  anchor_frame - PROPOSED: the action head sees only (a) a local crop centred on the grounded
                 point and (b) the end-effector position *relative to* that point. No global image,
                 no absolute coordinates -> translation-equivariant by construction.
  af_nocrop    - ablation: relative vector only, no local crop
  af_global    - ablation: anchor_frame + the global image features and absolute proprio back in
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

CROP = 24
DN = ("NHWC", "HWIO", "NHWC")
KINDS = ["bc", "bc_cotrain", "inject", "anchor_frame", "af_nocrop", "af_global"]


# ---------------------------------------------------------------- primitives
def _conv_p(key, k, cin, cout):
    return {"w": jax.random.normal(key, (k, k, cin, cout)) * np.sqrt(2.0 / (k * k * cin)),
            "b": jnp.zeros(cout)}


def _dense_p(key, din, dout, scale=1.0):
    return {"w": jax.random.normal(key, (din, dout)) * np.sqrt(2.0 / din) * scale,
            "b": jnp.zeros(dout)}


def conv(x, p, stride=1):
    return jax.lax.conv_general_dilated(x, p["w"], (stride, stride), "SAME",
                                        dimension_numbers=DN) + p["b"]


def dense(x, p):
    return x @ p["w"] + p["b"]


def _film_p(key, demb, c):
    k1, k2 = jax.random.split(key)
    return {"g": _dense_p(k1, demb, c, 0.1), "b": _dense_p(k2, demb, c, 0.1)}


def film(h, e, p):
    g = 1.0 + dense(e, p["g"])
    b = dense(e, p["b"])
    return h * g[:, None, None, :] + b[:, None, None, :]


# ---------------------------------------------------------------- language
def init_lang(key, d=16):
    k1, k2 = jax.random.split(key)
    return {"color": jax.random.normal(k1, (4, d)) * 0.5, "shape": jax.random.normal(k2, (3, d)) * 0.5}


def lang(p, cid, sid):
    # additive / concatenative composition: colour and shape tokens are embedded separately,
    # so a held-out colour x shape pair is still expressible.
    return jnp.concatenate([p["color"][cid], p["shape"][sid]], -1)


# ---------------------------------------------------------------- grounding net (pointing)
def init_ground(key):
    ks = jax.random.split(key, 8)
    return {"lang": init_lang(ks[0]),
            "c1": _conv_p(ks[1], 3, 3, 32), "f1": _film_p(ks[2], 32, 32),
            "c2": _conv_p(ks[3], 5, 32, 32), "f2": _film_p(ks[4], 32, 32),
            "c3": _conv_p(ks[5], 5, 32, 32), "f3": _film_p(ks[6], 32, 32),
            "c4": _conv_p(ks[7], 1, 32, 1)}


def ground_logits(p, img, cid, sid):
    """Fully convolutional, language-FiLMed heat-map over a 32x32 grid (translation-equivariant)."""
    e = lang(p["lang"], cid, sid)
    h = jax.nn.relu(film(conv(img, p["c1"], 2), e, p["f1"]))
    h = jax.nn.relu(film(conv(h, p["c2"]), e, p["f2"]))
    h = jax.nn.relu(film(conv(h, p["c3"]), e, p["f3"]))
    return conv(h, p["c4"])[..., 0]                       # (B,32,32)


_g = (jnp.arange(32) + 0.5) / 32.0


def logits_to_xy(logits, temp=1.0):
    B = logits.shape[0]
    pr = jax.nn.softmax(logits.reshape(B, -1) / temp, -1).reshape(B, 32, 32)
    x = (pr.sum(1) * _g).sum(-1)                           # columns -> x
    y = (pr.sum(2) * (1.0 - _g)).sum(-1)                   # rows -> y (up)
    return jnp.stack([x, y], -1)


def logits_to_xy_argmax(logits):
    B = logits.shape[0]
    idx = jnp.argmax(logits.reshape(B, -1), -1)
    r, c = idx // 32, idx % 32
    return jnp.stack([_g[c], 1.0 - _g[r]], -1)


# ---------------------------------------------------------------- policy blocks
GENC_MODE = "ssm"   # "ssm": spatial-softmax keypoints (robomimic-style); "flatten": 8x8 grid -> dense


def init_genc(key):
    ks = jax.random.split(key, 8)
    p = {"c1": _conv_p(ks[0], 3, 3, 32), "f1": _film_p(ks[1], 32, 32),
         "c2": _conv_p(ks[2], 3, 32, 64), "f2": _film_p(ks[3], 32, 64),
         "c3": _conv_p(ks[4], 3, 64, 64), "f3": _film_p(ks[5], 32, 64),
         "pt": _conv_p(ks[7], 1, 64, 1)}
    p["d"] = _dense_p(ks[6], 64 * 3 if GENC_MODE == "ssm" else 8 * 8 * 64, 256)
    return p


_g16 = (jnp.arange(16) + 0.5) / 16.0


def genc(p, img, e):
    h = jax.nn.relu(film(conv(img, p["c1"], 2), e, p["f1"]))     # 32x32
    h16 = jax.nn.relu(film(conv(h, p["c2"], 2), e, p["f2"]))     # 16x16
    if p["d"]["w"].shape[0] == 64 * 3:                            # spatial softmax keypoints
        h = film(conv(h16, p["c3"]), e, p["f3"])                  # 16x16x64 logits
        B, H, W, C = h.shape
        pr = jax.nn.softmax(h.reshape(B, H * W, C), axis=1).reshape(B, H, W, C)
        kx = (pr.sum(1) * _g16[None, :, None]).sum(1)             # (B,C) expected x
        ky = (pr.sum(2) * (1.0 - _g16)[None, :, None]).sum(1)     # (B,C) expected y
        feat = jnp.concatenate([kx * 2 - 1, ky * 2 - 1, jax.nn.relu(h).mean((1, 2))], -1)
    else:
        h = jax.nn.relu(film(conv(h16, p["c3"], 2), e, p["f3"]))  # 8x8
        feat = h.reshape(h.shape[0], -1)
    z = jax.nn.relu(dense(feat, p["d"]))
    point_logits = conv(h16, p["pt"])[..., 0]                    # (B,16,16) aux pointing head
    return z, point_logits


def init_cenc(key):
    ks = jax.random.split(key, 4)
    return {"c1": _conv_p(ks[0], 3, 3, 32), "c2": _conv_p(ks[1], 3, 32, 64),
            "c3": _conv_p(ks[2], 3, 64, 64), "d": _dense_p(ks[3], 6 * 6 * 64, 128)}


def cenc(p, crop):
    h = jax.nn.relu(conv(crop, p["c1"]))
    h = jax.nn.relu(conv(h, p["c2"], 2))
    h = jax.nn.relu(conv(h, p["c3"], 2))
    return jax.nn.relu(dense(h.reshape(h.shape[0], -1), p["d"]))


def extract_crop(img, anchor):
    """img (B,64,64,3) float, anchor (B,2) in workspace coords -> (B,CROP,CROP,3)."""
    half = CROP // 2
    pad = jnp.pad(img, ((0, 0), (half, half), (half, half), (0, 0)))
    row = jnp.clip(jnp.round((1.0 - anchor[:, 1]) * 64 - 0.5).astype(jnp.int32), 0, 63)
    col = jnp.clip(jnp.round(anchor[:, 0] * 64 - 0.5).astype(jnp.int32), 0, 63)
    # crop centred on (row, col): in padded coords the top-left corner is (row, col)
    return jax.vmap(lambda im, r, c: jax.lax.dynamic_slice(im, (r, c, 0), (CROP, CROP, 3)))(pad, row, col)


WRIST = ("bc", "bc_cotrain", "inject")      # policies with an end-effector-centred ("wrist") crop


def _head_in_dim(kind):
    return {"bc": 256 + 2 + 128, "bc_cotrain": 256 + 2 + 128, "inject": 256 + 2 + 64 + 128,
            "anchor_frame": 128 + 64, "af_nocrop": 64, "af_global": 256 + 2 + 128 + 64}[kind]


def init_policy(key, kind):
    ks = jax.random.split(key, 8)
    p = {"lang": init_lang(ks[0])}
    if kind in ("bc", "bc_cotrain", "inject", "af_global"):
        p["genc"] = init_genc(ks[1])
    if kind in ("anchor_frame", "af_global"):
        p["cenc"] = init_cenc(ks[2])
    if kind in WRIST:
        p["wenc"] = init_cenc(jax.random.fold_in(ks[2], 1))
    if kind == "inject":
        p["aemb"] = {"l1": _dense_p(ks[3], 2, 64), "l2": _dense_p(ks[4], 64, 64)}
    if kind in ("anchor_frame", "af_nocrop", "af_global"):
        p["remb"] = {"l1": _dense_p(ks[3], 2, 64), "l2": _dense_p(ks[4], 64, 64)}
    din = _head_in_dim(kind)
    p["head"] = {"l1": _dense_p(ks[5], din, 256), "l2": _dense_p(ks[6], 256, 256),
                 "l3": _dense_p(ks[7], 256, 3, 0.1)}
    return p


def _mlp2(x, p):
    return jax.nn.relu(dense(jax.nn.relu(dense(x, p["l1"])), p["l2"]))


def policy_apply(p, kind, img, ee, cid, sid, anchor):
    """img (B,64,64,3) float in [0,1]; ee, anchor (B,2) in [0,1]. Returns (a_xy, grip_logit, aux)."""
    e = lang(p["lang"], cid, sid)
    feats, aux = [], None
    if kind in ("bc", "bc_cotrain", "inject", "af_global"):
        z, aux = genc(p["genc"], img, e)
        feats += [z, ee * 2 - 1]
    if kind in WRIST:
        feats.append(cenc(p["wenc"], extract_crop(img, ee)))
    if kind == "inject":
        feats.append(_mlp2(anchor * 2 - 1, p["aemb"]))
    if kind in ("anchor_frame", "af_global"):
        feats.append(cenc(p["cenc"], extract_crop(img, anchor)))
    if kind in ("anchor_frame", "af_nocrop", "af_global"):
        feats.append(_mlp2((ee - anchor) * 3.0, p["remb"]))
    x = jnp.concatenate(feats, -1)
    h = jax.nn.relu(dense(x, p["head"]["l1"]))
    h = jax.nn.relu(dense(h, p["head"]["l2"]))
    out = dense(h, p["head"]["l3"])
    return jnp.tanh(out[:, :2]), out[:, 2], aux


def uses_anchor(kind):
    return kind in ("inject", "anchor_frame", "af_nocrop", "af_global")
