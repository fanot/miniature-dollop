"""All TempoBench experiments. Results -> results_tempo/*.json

    python scripts/tempo_experiments.py --seeds 0 1 2

E1  mechanism: executed/planned speed on long cruises (oracle planner) vs the formula
E2  speed curves of the learned policy: score / success vs playback speed, robot lag, frozen vs
    expiring plans, SAIL-style adaptive speed, arm-only time shift
E3  'ft1' analogue: fine-tune the scripted policy on slower teleop-style demos; open-loop error
    vs planned tempo vs closed-loop score over a family of checkpoints
E4  calibration: speed predicted from measured quantities vs the best speed found by search
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tempobench import closed_loop as C   # noqa: E402
from tempobench import feed as F          # noqa: E402
from tempobench import policy as P        # noqa: E402
from tempobench import world as W         # noqa: E402

LAGS = [0.08, 0.13, 0.18]
SPEEDS = [1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.4]
TELEOP = dict(tempo=0.72, omega=W.omega_for_lag(0.13), p_pause=0.35, tempo_jitter=0.25)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ---------------------------------------------------------------------------- E1
def long_world_factory():
    orig = W.World.__init__

    def init(self, n, rng, omega=W.OMEGA_DEMO, timeouts=W.T_SUB):
        orig(self, n, rng, omega, timeouts)
        self.p[:] = np.c_[np.full(n, 0.03), rng.uniform(0.3, 0.7, n)]
        self.obj[:] = np.c_[np.full(n, 3.0), self.p[:, 1]]
        self.place[:] = np.c_[np.full(n, 0.03), self.p[:, 1]]
    return orig, init


def e1(out):
    orig, init = long_world_factory()
    W.World.__init__ = init
    rows = []
    try:
        for ds in [0.08, 0.13, 0.18, 0.23]:
            for s in [0.8, 1.0, 1.15, 1.3, 1.5]:
                for life in [1.2, 0.6]:
                    cfg = F.FeedConfig(speed=s, blend_life=life, vlim=1e9, alim=1e9)
                    r = C.rollout(lambda w: P.oracle_plan(w), cfg, n=60, omega=W.omega_for_lag(ds),
                                  timeouts=(30, 30), max_t=4.0)
                    rows.append(dict(delta_sim=ds, speed=s, blend_life=life, measured=r["tempo_ratio_abs"],
                                     predicted=F.predicted_tempo_ratio(cfg, ds)))
                    print(f"  E1 ds={ds} s={s} life={life} meas={rows[-1]['measured']:.3f} "
                          f"pred={rows[-1]['predicted']:.3f}", flush=True)
    finally:
        W.World.__init__ = orig
    jdump(rows, os.path.join(out, "e1_formula.json"))


# ---------------------------------------------------------------------------- data / models
def datasets():
    _, rs = W.run_expert(1500, seed=1)
    _, rt = W.run_expert(800, seed=2, **TELEOP)
    _, rsv = W.run_expert(200, seed=99)
    _, rtv = W.run_expert(200, seed=98, **TELEOP)
    d = {}
    for k, r in [("scripted", rs), ("teleop", rt), ("scripted_val", rsv), ("teleop_val", rtv)]:
        X, Y, _ = W.make_chunks(r)
        d[k] = (X, Y)
    return d


def models_for_seed(seed, d, ckdir, steps=8000, ft_steps=3000):
    """base_2k / base_4k / base (scripted only) and ft_25 / ft_50 / ft_100 (fine-tuned from base on
    a mix with 25 / 50 / 100 % teleop frames, like ft1: teleop scenes + scripted at lower weight)."""
    Xs, Ys = d["scripted"]
    Xt, Yt = d["teleop"]
    out = {}
    path = lambda k: os.path.join(ckdir, f"{k}_s{seed}.pkl")  # noqa: E731
    if all(os.path.exists(path(k)) for k in ["base_2k", "base_4k", "base", "ft_25", "ft_50", "ft_100"]):
        return {k: P.load(path(k)) for k in ["base_2k", "base_4k", "base", "ft_25", "ft_50", "ft_100"]}
    p = P.train(Xs, Ys, seed=seed, steps=2000, log_every=0); out["base_2k"] = p
    p4 = P.train(Xs, Ys, seed=seed, steps=4000, log_every=0); out["base_4k"] = p4
    pb = P.train(Xs, Ys, seed=seed, steps=steps, log_every=4000, tag=f" s{seed}"); out["base"] = pb
    rng = np.random.RandomState(seed)
    for frac in (25, 50, 100):
        nt = len(Xt)
        ns = int(nt * (100 - frac) / max(frac, 1))
        i = rng.choice(len(Xs), min(ns, len(Xs)), replace=False)
        X = np.concatenate([Xt, Xs[i]])
        Y = np.concatenate([Yt, Ys[i]])
        out[f"ft_{frac}"] = P.train(X, Y, seed=seed + 50 + frac, steps=ft_steps, lr=2e-4, init=pb,
                                    log_every=0)
    for k, v in out.items():
        P.save(v, path(k))
    return out


# ---------------------------------------------------------------------------- E2
def e2(seed, pol, out, timeouts, n=300):
    res = []
    for ds in LAGS:
        om = W.omega_for_lag(ds)
        for variant, base_cfg in [("uniform", F.FeedConfig()), ("uniform+expire", F.FeedConfig(expire=True))]:
            for s in SPEEDS:
                r = C.rollout(pol, dataclasses.replace(base_cfg, speed=s), n=n, seed=seed, omega=om, timeouts=timeouts)
                res.append(dict(delta_sim=ds, variant=variant, knob=s, **r))
        for a in [1.25, 1.5, 2.0, 2.5]:
            r = C.rollout(pol, F.FeedConfig(adaptive_speed=a), n=n, seed=seed, omega=om, timeouts=timeouts)
            res.append(dict(delta_sim=ds, variant="adaptive", knob=a, **r))
        for sh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            r = C.rollout(pol, F.FeedConfig(arm_shift=sh), n=n, seed=seed, omega=om, timeouts=timeouts)
            res.append(dict(delta_sim=ds, variant="arm_shift", knob=sh, **r))
        uni = [x for x in res if x["delta_sim"] == ds and x["variant"] == "uniform"]
        best = max(uni, key=lambda x: x["score"])
        print(f"  E2 seed {seed} ds={ds}: s=1 score {uni[0]['score']:.2f}, "
              f"best uniform s={best['knob']} score={best['score']:.2f}", flush=True)
    jdump(res, os.path.join(out, f"e2_speed_s{seed}.json"))
    return res


# ---------------------------------------------------------------------------- E3 / E4
def e3(seed, models, d, out, timeouts, n=300):
    Xsv, Ysv = d["scripted_val"]
    Xtv, Ytv = d["teleop_val"]
    rows = []
    for name, params in models.items():
        pol = P.FlowPolicy(params, seed=seed)
        r = dict(model=name)
        r["err_teleop_val"] = P.open_loop_error(pol, Xtv, Ytv, seed=seed)
        r["err_scripted_val"] = P.open_loop_error(pol, Xsv, Ysv, seed=seed)
        tp, td = P.planned_tempo(pol, Xsv, Ysv, seed=seed)
        r["planned_tempo"] = tp / td
        cl = C.rollout(pol, F.FeedConfig(), n=n, seed=seed, timeouts=timeouts)
        r.update({f"cl_{k}": v for k, v in cl.items()})
        # E4: calibrated speed = feed compensation at matched lag / the model's own tempo
        s_feed = F.predicted_best_speed(F.FeedConfig(), W.lag_of(W.OMEGA_DEMO))
        r["s_cal"] = s_feed / max(r["planned_tempo"], 1e-3)
        cc = C.rollout(pol, F.FeedConfig(speed=r["s_cal"]), n=n, seed=seed, timeouts=timeouts)
        r.update({f"cal_{k}": v for k, v in cc.items()})
        sweep = []
        for s in [1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0]:
            cs = C.rollout(pol, F.FeedConfig(speed=s), n=n // 2, seed=seed + 7, timeouts=timeouts)
            sweep.append((s, cs["score"], cs["subtask_success"]))
        r["sweep"] = sweep
        rows.append(r)
        print(f"  E3 seed {seed} {name:8s} err_tel={r['err_teleop_val']:.4f} err_scr={r['err_scripted_val']:.4f} "
              f"tempo={r['planned_tempo']:.2f} score@1={r['cl_score']:.2f} succ@1={r['cl_subtask_success']:.2f} "
              f"s_cal={r['s_cal']:.2f} score@cal={r['cal_score']:.2f}", flush=True)
    jdump(rows, os.path.join(out, f"e3_models_s{seed}.json"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results_tempo")
    ap.add_argument("--skip_e1", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "ckpt")
    os.makedirs(ck, exist_ok=True)
    t0 = time.time()
    timeouts = W.calibrate_timeouts()
    jdump({"timeouts": timeouts, "teleop": TELEOP, "feed_default": dataclasses.asdict(F.FeedConfig()),
           "mean_plan_age": [float((w * a_).sum() / w.sum()) for a_, _, w in F.mean_plan_age(F.FeedConfig())][:1],
           "s_star_pred": {str(ds): F.predicted_best_speed(F.FeedConfig(), ds) for ds in LAGS}},
          os.path.join(a.out, "config.json"))
    if not a.skip_e1:
        e1(a.out)
        print(f"E1 done ({time.time()-t0:.0f}s)", flush=True)
    d = datasets()
    print("data:", {k: v[0].shape[0] for k, v in d.items()}, flush=True)
    for seed in a.seeds:
        models = models_for_seed(seed, d, ck)
        print(f"seed {seed}: models ready ({time.time()-t0:.0f}s)", flush=True)
        e3(seed, models, d, a.out, timeouts, n=a.n)
        e2(seed, P.FlowPolicy(models["base"], seed=seed), a.out, timeouts, n=a.n)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------- E1b (run separately)
def e1b(out, ckpt=None, seed=0, n=300):
    """Blend on/off and blend life on the task: oracle and (if given) the learned base policy."""
    timeouts = W.calibrate_timeouts()
    planners = {"oracle": lambda w: P.oracle_plan(w)}
    if ckpt:
        planners["learned"] = P.FlowPolicy(P.load(ckpt), seed=seed)
    rows = []
    for name, pl in planners.items():
        for ds in LAGS:
            for blend, life in [(False, 0.0), (True, 0.4), (True, 0.8), (True, 1.2)]:
                cfg = F.FeedConfig(blend=blend, blend_life=life if blend else 1.2)
                r = C.rollout(pl, cfg, n=n, seed=seed, omega=W.omega_for_lag(ds), timeouts=timeouts)
                rows.append(dict(planner=name, delta_sim=ds, blend=blend, life=life, **r))
                print(f"  E1b {name} ds={ds} blend={blend} life={life}: score={r['score']:.2f} "
                      f"succ={r['subtask_success']:.2f} tempo={r['tempo_ratio_abs']:.2f} clamp={r['clamped_frac']:.2f}",
                      flush=True)
    jdump(rows, os.path.join(out, "e1b_blend.json"))
