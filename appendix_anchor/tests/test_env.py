"""Sanity checks: the scripted expert solves every split through the same closed loop the policies use,
and the anchor-frame crop really is centred on the anchor."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anchorbench import env as E  # noqa: E402
from anchorbench import evaluate as V  # noqa: E402


def _expert_rollout(scenes):
    A = V.scenes_to_arrays(scenes)

    def fake_policy(kind):
        def f(p, img, ee, cid, sid, anchor):
            out = [E.expert_action(np.asarray(ee[n]), A["pos"][n, 0], A["theta"][n, 0]) for n in range(len(ee))]
            return np.array([o[0] for o in out]), np.array([float(o[1]) for o in out])
        return f

    orig = V._policy_jit
    V._policy_jit = fake_policy
    try:
        return V.rollout("bc", None, scenes, anchor_mode="oracle")
    finally:
        V._policy_jit = orig


def test_expert_solves_all_splits():
    rng = np.random.RandomState(0)
    for split in ["id", "pos_ood", "combo_ood", "swap", "bg"]:
        scenes = [E.sample_scene(rng, split) for _ in range(60)]
        r = _expert_rollout(scenes)
        assert (r["outcome"] == 0).all(), split


def test_empty_split_has_no_target():
    rng = np.random.RandomState(1)
    s = E.sample_scene(rng, "empty")
    assert not s.present[0] and s.present[1:].all()


def test_swap_moves_target_to_distractor_place():
    a = E.sample_scene(np.random.RandomState(2), "swap")
    # the named object now lies near the usual place of the distractor in slot 1
    assert np.linalg.norm(a.pos[0] - E.ANCHORS[a.ids[1]]) < 0.2


def test_crop_is_centred():
    import jax.numpy as jnp
    from anchorbench import models as M
    img = np.zeros((1, 64, 64, 3), np.float32)
    img[0, 20, 40] = 1.0                      # pixel row 20, col 40
    anchor = np.array([[(40 + 0.5) / 64, 1 - (20 + 0.5) / 64]], np.float32)
    crop = np.asarray(M.extract_crop(jnp.asarray(img), jnp.asarray(anchor)))
    assert crop[0, M.CROP // 2, M.CROP // 2, 0] == 1.0
