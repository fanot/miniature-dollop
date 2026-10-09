"""Measure the closed-loop tempo deficit of the served Green Challenge policy.

Uses only what the green_challenge repo already produces:
  * the WBC dataset (demonstrations)            -> demo joint speeds and the action->state lead
  * GC_RECORD_DIR recordings of bench rollouts   -> closed-loop joint speeds (state every 3rd call)

    # 1) demonstrations of the evaluation scenes (any LeRobot v2.1 root that gch.wbc reads)
    python greensim/tempo_probe.py demo --data <scene root> [--episodes-json ring_eps.json] --out demo.json
    # 2) closed loop: bench run of the baseline with recording on (see bench/README.md, si/README.md)
    GC_RECORD_DIR=rec_base SEEDS=42 VARIANTS=bench/variants_baseline.txt bash bench/run.sh ~/green_sim full
    python greensim/tempo_probe.py sim --rec rec_base --out sim_base.json
    # 3) compare, predict the playback speed that removes the deficit
    python greensim/tempo_probe.py compare demo.json sim_base.json --speed 1.0

NOT RUN in this repository: there is no dataset or simulator here. The demo reader is written
against gch.wbc.WBCDataset (load_episode / split / Episode.subtask_at) and gch.inspect_wbc.best_lag
from my green_challenge code; put that repo on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# raw-51 state layout (gch.layout.RAW_GROUPS)
GROUPS = {"left_arm": slice(13, 18), "right_arm": slice(18, 23)}
FPS = 30.0
REC_DT = 0.08 * 3          # recorder keeps every 3rd call, calls are 0.08 s apart
MOVING = 0.15              # rad/s: frames slower than this are "not moving" (holds, grasps)

# feed defaults of the served policy (organisers' action_feed.py + 1/30 s rows)
LEAD, PERIOD, LIFE, TAU, ROW = 0.030, 0.080, 1.20, 1.00, 1.0 / 30.0


def mean_plan_age(speed=1.0):
    row = ROW / speed
    ages = []
    for f in np.linspace(0, PERIOD, 200, endpoint=False):
        a = LEAD + row + f + PERIOD * np.arange(40)
        a = a[(a <= LIFE + 1e-9) | (np.arange(40) == 0)]
        w = np.exp(-a / TAU)
        ages.append((w * a).sum() / w.sum())
    return float(np.mean(ages))


def speeds(q, dt, window):
    """Joint-space speed |q[t+window]-q[t]| / (window*dt) for one group, all frames."""
    if len(q) <= window:
        return np.zeros(0)
    return np.linalg.norm(q[window:] - q[:-window], axis=1) / (window * dt)


def summarize(v):
    v = np.asarray(v)
    m = v[v > MOVING]
    if len(m) == 0:
        return {"n": 0}
    return {"n": int(len(m)), "median": float(np.median(m)), "p75": float(np.quantile(m, 0.75)),
            "p90": float(np.quantile(m, 0.9)), "moving_frac": float(len(m) / len(v))}


def cmd_demo(a):
    from gch.inspect_wbc import best_lag
    from gch.wbc import WBCDataset

    ds = WBCDataset(a.data)
    if a.episodes_json:
        episodes = json.load(open(a.episodes_json))       # e.g. the held-out list from fetch_data.py
    else:
        episodes = ds.split("train")
    window = int(round(REC_DT * FPS))      # same spacing as the recorder
    per = {g: [] for g in GROUPS}
    lead = {g: [] for g in GROUPS}
    for ep in episodes[: a.max_episodes]:
        d = ds.load_episode(ep)
        for g, sl in GROUPS.items():
            per[g].append(speeds(d["state"][:, sl], 1.0 / FPS, window))
            lead[g].append(best_lag(d["state"][:, sl], d["action"][:, sl])[0])
    out = {g: summarize(np.concatenate(v)) for g, v in per.items() if v}
    for g in out:
        out[g]["action_lead_frames"] = float(np.median(lead[g]))
        out[g]["delta_demo_s"] = out[g]["action_lead_frames"] / FPS
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_sim(a):
    per = {g: [] for g in GROUPS}
    for seg in sorted(Path(a.rec).glob("seg_*")):
        files = sorted(seg.glob("*.npz"))
        if len(files) < 3:
            continue
        recs = [np.load(f) for f in files]
        t = np.array([float(r["t"]) for r in recs])
        st = np.stack([r["state"] for r in recs]).astype(np.float64)
        for g, sl in GROUPS.items():
            dq = np.linalg.norm(np.diff(st[:, sl], axis=0), axis=1)
            dt = np.diff(t)
            ok = dt > 1e-6
            per[g].append(dq[ok] / dt[ok])
    out = {g: summarize(np.concatenate(v)) for g, v in per.items() if v}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


def cmd_compare(a):
    demo, sim = json.load(open(a.demo)), json.load(open(a.sim))
    s = a.speed
    abar = mean_plan_age(s)
    rep = {"mean_plan_age_s": abar, "speed": s}
    for g in GROUPS:
        if g not in demo or g not in sim or not demo[g].get("n") or not sim[g].get("n"):
            continue
        ratio = sim[g]["median"] / demo[g]["median"]        # executed / demonstrated
        dd = demo[g]["delta_demo_s"]
        # invert u/v = (dd + s*(abar - lead)) / (abar + ds) for the deployment lag ds
        ds_ = (dd + s * (abar - LEAD)) / max(ratio, 1e-6) - abar
        s_formula = (mean_plan_age(1.0) + ds_ - dd) / (mean_plan_age(1.0) - LEAD)
        rep[g] = {"tempo_ratio": ratio, "delta_demo_s": dd, "delta_sim_inferred_s": ds_,
                  "speed_to_close_deficit_formula": s_formula,
                  "speed_to_close_deficit_linear": s / max(ratio, 1e-6),
                  "frozen_plans_from_speed": (ROW * 50) / (LIFE - LEAD)}
    json.dump(rep, open(a.out, "w"), indent=1) if a.out else None
    print(json.dumps(rep, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--data", required=True)
    d.add_argument("--episodes-json", default="", help="JSON list of episode indices of the evaluation scene")
    d.add_argument("--max-episodes", type=int, default=300)
    d.add_argument("--out", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--rec", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("demo")
    c.add_argument("sim")
    c.add_argument("--speed", type=float, default=1.0)
    c.add_argument("--out", default="")
    a = p.parse_args()
    {"demo": cmd_demo, "sim": cmd_sim, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
