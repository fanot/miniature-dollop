"""Follow-up experiments on the checkpoints of tempo_experiments.py.

    python scripts/tempo_extra.py --seeds 0 1 2

E1b  blend on/off and blend life on the task (oracle + learned)
E2b  adaptive speed with two gripper-event detectors:
       jitter   - any row-to-row gripper change > 0.02 (the logic of my green_challenge patch, which flags
                  finger travel > 0.01 rad between rows);
       crossing - rows where the plan's gripper crosses 0.5 (a real open/close);
     plus how many rows each detector flags on demonstration chunks vs on sampled plans
E2c  share of the deficit caused by the acceleration limiter: clamp on vs off
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tempobench import closed_loop as C   # noqa: E402
from tempobench import feed as F          # noqa: E402
from tempobench import policy as P        # noqa: E402
from tempobench import world as W         # noqa: E402
from tempo_experiments import LAGS, e1b, jdump  # noqa: E402


def detector_stats(pol, seed):
    _, rv = W.run_expert(200, seed=99)
    Xv, Yv, _ = W.make_chunks(rv)
    rng = np.random.RandomState(seed)
    i = rng.choice(len(Xv), 3000, replace=False)
    out = {}
    for name, rows in [("demo_chunks", Yv[i]), ("policy_plans", pol.sample(Xv[i]))]:
        g = rows[:, :, 2]
        jit = np.abs(np.diff(g, axis=1)) > 0.02
        cross = np.diff((g > 0.5).astype(int), axis=1) != 0
        out[name] = {"rows_flagged_jitter": float(jit.mean()), "rows_flagged_crossing": float(cross.mean()),
                     "chunks_with_jitter_event": float(jit.any(1).mean()),
                     "chunks_with_crossing_event": float(cross.any(1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--res", default="results_tempo")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    timeouts = W.calibrate_timeouts()
    e1b(a.res, ckpt=os.path.join(a.res, "ckpt", "base_s0.pkl"), seed=0, n=a.n)
    for seed in a.seeds:
        pol = P.FlowPolicy(P.load(os.path.join(a.res, "ckpt", f"base_s{seed}.pkl")), seed=seed)
        rows = []
        for ds in LAGS:
            om = W.omega_for_lag(ds)
            for det in ("jitter", "crossing"):
                for sp in (1.5, 2.0, 2.5, 3.0):
                    r = C.rollout(pol, F.FeedConfig(adaptive_speed=sp, adaptive_detector=det), n=a.n, seed=seed,
                                  omega=om, timeouts=timeouts)
                    rows.append(dict(delta_sim=ds, variant=f"adaptive_{det}", knob=sp, **r))
            for sp in (1.0, 1.3):
                for clamp in (True, False):
                    cfg = F.FeedConfig(speed=sp) if clamp else F.FeedConfig(speed=sp, vlim=1e9, alim=1e9)
                    r = C.rollout(pol, cfg, n=a.n, seed=seed, omega=om, timeouts=timeouts)
                    rows.append(dict(delta_sim=ds, variant="clamp_on" if clamp else "clamp_off", knob=sp, **r))
            best = {d: max((x for x in rows if x["delta_sim"] == ds and x["variant"] == f"adaptive_{d}"),
                           key=lambda x: x["score"]) for d in ("jitter", "crossing")}
            print(f"  seed {seed} ds={ds}: adaptive jitter best {best['jitter']['score']:.2f} "
                  f"(slow rows {best['jitter']['slow_rows_frac']:.2f}), crossing best {best['crossing']['score']:.2f} "
                  f"@{best['crossing']['knob']} (slow rows {best['crossing']['slow_rows_frac']:.2f})", flush=True)
        jdump({"rows": rows, "detector": detector_stats(pol, seed)}, os.path.join(a.res, f"e2b_adaptive_s{seed}.json"))


if __name__ == "__main__":
    main()
