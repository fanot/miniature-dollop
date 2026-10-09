"""Full AnchorBench experiment for one or more seeds.

    python scripts/run_experiment.py --seeds 0 1 2 --out results

Per seed: generate narrow robot demos + broad pointing data, train the pointing net and all
policies, then run the closed-loop splits and the causal probes. Everything is written to
results/seed_<k>.json (+ a few trajectories in results/traj_seed_<k>.npz).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anchorbench import env as E          # noqa: E402
from anchorbench import data as D         # noqa: E402
from anchorbench import train as T        # noqa: E402
from anchorbench import evaluate as V     # noqa: E402

SPLITS = ["id", "pos_ood", "combo_ood", "swap", "bg", "empty"]
GUIDE_LAMBDA = 0.5


def eval_scenes(split, n, seed=12345):
    rng = np.random.RandomState(seed + SPLITS.index(split) * 1000)
    return [E.sample_scene(rng, split) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_demos", type=int, default=600)
    ap.add_argument("--n_web", type=int, default=12000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ground_steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--p_uniform", type=float, default=0.0,
                    help="fraction of demos with uniformly random layouts (data-diversity sweep)")
    ap.add_argument("--methods", nargs="+",
                    default=["bc", "bc_cotrain", "inject", "anchor_frame", "af_nocrop", "af_global"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)

    scenes = {s: eval_scenes(s, args.eval_n) for s in SPLITS}

    for seed in args.seeds:
        t0 = time.time()
        print(f"=== seed {seed} ===", flush=True)
        robot = D.generate_demos(args.n_demos, seed=seed, p_uniform=args.p_uniform)
        web = D.generate_pointing(args.n_web, seed=seed)
        print(f"  data: {len(robot['img'])} robot frames from {args.n_demos} demos, "
              f"{len(web['img'])} pointing images ({time.time()-t0:.0f}s)", flush=True)
        out = {"config": vars(args), "seed": seed, "data": {
            "robot_frames": int(len(robot["img"])),
            "train_target_xy": robot["tgt_xy"][np.r_[True, robot["ep"][1:] != robot["ep"][:-1]]].tolist(),
        }, "splits": {}, "probes": {}}

        ck = lambda k: os.path.join(args.out, "ckpt", f"{k}_s{seed}{args.tag}.pkl")  # noqa: E731
        if os.path.exists(ck("ground")):                      # resume after an interrupted run
            gp = T.load(ck("ground"))
        else:
            gp = T.train_grounding(robot, web, seed, steps=args.ground_steps)
            T.save(gp, ck("ground"))
        params = {}
        for k in args.methods:
            if os.path.exists(ck(k)):
                params[k] = T.load(ck(k))
                continue
            params[k] = T.train_policy(k, robot, web, seed, steps=args.steps, bs=args.bs)
            T.save(params[k], ck(k))
        del robot, web

        runs = [(k, k, 0.0) for k in args.methods]
        if "bc_cotrain" in params:
            runs.insert(2, ("guide", "bc_cotrain", GUIDE_LAMBDA))
        trajs = {}
        for name, kind, lam in runs:
            out["splits"][name] = {}
            for s in SPLITS:
                r = V.rollout(kind, params[kind], scenes[s], gparams=gp, guide_lambda=lam, seed=seed)
                out["splits"][name][s] = V.summarize(r)
                if s in ("swap", "pos_ood", "empty"):
                    trajs[f"{name}/{s}"] = r["traj"][:12]
            sr = {s: round(out["splits"][name][s]["success"], 3) for s in SPLITS if s != "empty"}
            print(f"  {name:13s} {sr}  air@usual={out['splits'][name]['empty']['air_at_usual_place']:.2f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        # ---------------- causal probes on anchor-using methods
        anchor_runs = [(n, k, lam) for n, k, lam in runs if n in ("guide", "inject", "anchor_frame", "af_global", "af_nocrop")]
        for name, kind, lam in anchor_runs:
            pr = {}
            for s in ("pos_ood", "combo_ood"):
                r = V.rollout(kind, params[kind], scenes[s], gparams=gp, anchor_mode="oracle",
                              guide_lambda=lam, seed=seed)
                pr[f"oracle_{s}"] = V.summarize(r)
            # counterfactual anchor: point at distractor #1 while the instruction still names the target
            sc = scenes["pos_ood"]
            ov = np.stack([x.pos[1] for x in sc])
            r = V.rollout(kind, params[kind], sc, gparams=gp, anchor_mode="override", override_xy=ov,
                          guide_lambda=lam, seed=seed)
            pr["counterfactual"] = {"follows_anchor": float(np.mean((r["outcome"] == 1) & (r["grabbed"] == 1))),
                                    "follows_language": float(np.mean(r["outcome"] == 0)),
                                    **V.summarize(r)}
            # systematic grounding error: persistent offset of the anchor, oracle otherwise
            for sig in (0.02, 0.04, 0.06, 0.08):
                r = V.rollout(kind, params[kind], sc, gparams=gp, anchor_mode="oracle",
                              anchor_noise=sig, guide_lambda=lam, seed=seed)
                pr[f"noise_{sig}"] = V.summarize(r)
            out["probes"][name] = pr
            print(f"  probe {name:13s} oracle_pos={pr['oracle_pos_ood']['success']:.2f} "
                  f"follow={pr['counterfactual']['follows_anchor']:.2f} "
                  f"noise.04={pr['noise_0.04']['success']:.2f} ({time.time()-t0:.0f}s)", flush=True)

        with open(os.path.join(args.out, f"seed_{seed}{args.tag}.json"), "w") as f:
            json.dump(out, f, indent=1)
        np.savez_compressed(os.path.join(args.out, f"traj_seed_{seed}{args.tag}.npz"), **trajs,
                            **{f"scene_{s}": np.array([[*x.pos.ravel(), *x.theta, *x.ids, x.target_id, *x.present, *x.ee0, *x.bg]
                                                       for x in scenes[s][:12]]) for s in ("swap", "pos_ood", "empty")})
        print(f"=== seed {seed} done in {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
