"""Two follow-ups that use the checkpoints of run_experiment.py.

1. Narrow pointer: if the pointing model is trained on the robot demos only (no broad pointing
   data), does the layout shortcut simply move into the pointer?  (README section 3, caveat)
2. Abstention: can the pointer's own confidence tell "the named object is not on the table"
   (split `empty`) from normal scenes, so that the anchor-frame policy could refuse to act?

    python scripts/run_pointer_ablation.py --seeds 0 1 2 --res results
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from anchorbench import data as D        # noqa: E402
from anchorbench import env as E         # noqa: E402
from anchorbench import evaluate as V    # noqa: E402
from anchorbench import models as M      # noqa: E402
from anchorbench import train as T       # noqa: E402
from run_experiment import eval_scenes   # noqa: E402


@jax.jit
def _peak_prob(gp, img, cid, sid):
    lg = M.ground_logits(gp, img.astype(jnp.float32) / 255.0, cid, sid)
    return jax.nn.softmax(lg.reshape(lg.shape[0], -1), -1).max(-1)


def confidence(gp, scenes):
    img = E.render_scenes(scenes)
    cid = np.array([E.IDENTS[s.target_id][0] for s in scenes])
    sid = np.array([E.IDENTS[s.target_id][1] for s in scenes])
    return np.asarray(_peak_prob(gp, img, cid, sid))


def auc(pos, neg):
    """P(conf of a scene with the target > conf of a scene without it)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--res", default="results")
    ap.add_argument("--eval_n", type=int, default=200)
    a = ap.parse_args()
    for seed in a.seeds:
        cfg = json.load(open(os.path.join(a.res, f"seed_{seed}.json")))["config"]
        robot = D.generate_demos(cfg["n_demos"], seed=seed, p_uniform=cfg["p_uniform"])
        gp_narrow = T.train_grounding(robot, robot, seed + 500, steps=cfg["ground_steps"])
        gp_broad = T.load(os.path.join(a.res, "ckpt", f"ground_s{seed}.pkl"))
        out = {"narrow_pointer": {}, "abstention": {}}
        for kind in ("anchor_frame", "inject"):
            params = T.load(os.path.join(a.res, "ckpt", f"{kind}_s{seed}.pkl"))
            out["narrow_pointer"][kind] = {}
            for s in ("id", "pos_ood", "combo_ood", "swap"):
                r = V.rollout(kind, params, eval_scenes(s, a.eval_n), gparams=gp_narrow, seed=seed)
                out["narrow_pointer"][kind][s] = V.summarize(r)
            print(f"  seed {seed} narrow pointer {kind}: " +
                  str({s: round(v['success'], 2) for s, v in out['narrow_pointer'][kind].items()}) +
                  f" pointing acc pos_ood={out['narrow_pointer'][kind]['pos_ood']['grounding_acc']:.2f}", flush=True)
        # abstention from pointer confidence (broad pointer)
        c_ok = confidence(gp_broad, eval_scenes("pos_ood", a.eval_n))
        c_empty = confidence(gp_broad, eval_scenes("empty", a.eval_n))
        tau = float(np.quantile(c_ok, 0.05))            # keep 95% of normal scenes
        out["abstention"] = {"auc": auc(c_ok, c_empty), "tau_keep95": tau,
                             "empty_refused": float(np.mean(c_empty < tau)),
                             "conf_ok_median": float(np.median(c_ok)), "conf_empty_median": float(np.median(c_empty))}
        print(f"  seed {seed} abstention: {out['abstention']}", flush=True)
        json.dump(out, open(os.path.join(a.res, f"pointer_ablation_seed_{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
