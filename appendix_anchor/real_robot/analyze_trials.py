"""Analyse a filled-in trial sheet from the real-robot protocol.

    python real_robot/analyze_trials.py --trials trials.csv \
        --train_positions my_demo_positions.csv --out real_robot/report

Produces report/summary.md, report/sr_vs_distance.png and report/position_heatmap.png.
Metrics are the same as in AnchorBench, so simulation and robot numbers can be put side by side.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--train_positions", required=True)
    ap.add_argument("--out", default="real_robot/report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    train = defaultdict(list)
    with open(a.train_positions) as f:
        for r in csv.DictReader(f):
            train[r["object"]].append((float(r["x_cm"]), float(r["y_cm"])))
    rows = [r for r in csv.DictReader(open(a.trials)) if r.get("outcome")]
    if not rows:
        raise SystemExit("no filled-in trials (column 'outcome' is empty everywhere)")

    lines = ["# Real-robot grounding-bypass report", "", f"Filled trials: {len(rows)}", ""]
    by_probe = defaultdict(list)
    for r in rows:
        by_probe[r["probe"]].append(r)

    lines += ["| probe | n | success | wrong object | miss | collision | timeout | air at usual place |",
              "|---|---|---|---|---|---|---|---|"]
    for probe, rs in sorted(by_probe.items()):
        n = len(rs)
        cnt = defaultdict(int)
        for r in rs:
            cnt[r["outcome"].strip()] += 1
        cells = []
        for o in ["success", "wrong_object", "miss", "collision", "timeout", "air_at_usual_place"]:
            p, lo, hi = wilson(cnt[o], n)
            cells.append(f"{p:.2f} [{lo:.2f},{hi:.2f}]")
        lines.append(f"| {probe} | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- P1: success vs distance to the nearest demonstrated position of the same object
    grid = by_probe.get("position_grid", [])
    if grid:
        d, s, xy = [], [], []
        for r in grid:
            tx, ty = float(r["target_x_cm"]), float(r["target_y_cm"])
            pts = np.array(train[r["target"]]) if train[r["target"]] else np.zeros((1, 2)) + 1e9
            d.append(np.min(np.linalg.norm(pts - [tx, ty], axis=1)))
            s.append(r["outcome"].strip() == "success")
            xy.append((tx, ty))
        d, s, xy = np.array(d), np.array(s), np.array(xy)
        bins = np.quantile(d, np.linspace(0, 1, min(5, len(d)) + 1))
        bins[-1] += 1e-6
        centers, rates, los, his = [], [], [], []
        for lo_b, hi_b in zip(bins[:-1], bins[1:]):
            m = (d >= lo_b) & (d < hi_b)
            if m.sum() == 0:
                continue
            p, lo, hi = wilson(int(s[m].sum()), int(m.sum()))
            centers.append(d[m].mean()); rates.append(p); los.append(p - lo); his.append(hi - p)
        plt.figure(figsize=(4.5, 3.2))
        plt.errorbar(centers, rates, yerr=[los, his], marker="o", capsize=3)
        plt.xlabel("distance to nearest demo position of this object, cm")
        plt.ylabel("success rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "sr_vs_distance.png"), dpi=150)
        plt.close()
        lines += ["## P1 position grid", "", "![](sr_vs_distance.png)", ""]
        # heat-map per cell
        cells = defaultdict(list)
        for (x, y), ok in zip(xy, s):
            cells[(x, y)].append(ok)
        xs = sorted({k[0] for k in cells}); ys = sorted({k[1] for k in cells})
        M = np.full((len(ys), len(xs)), np.nan)
        for (x, y), v in cells.items():
            M[ys.index(y), xs.index(x)] = np.mean(v)
        plt.figure(figsize=(4.5, 3.2))
        plt.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis",
                   extent=[0, xs[-1] + xs[0], 0, ys[-1] + ys[0]])
        for obj, pts in train.items():
            pts = np.array(pts)
            plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.6, label=f"demos: {obj}")
        plt.colorbar(label="success rate")
        plt.legend(fontsize=6, loc="upper right")
        plt.xlabel("x, cm"); plt.ylabel("y, cm")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "position_heatmap.png"), dpi=150)
        plt.close()
        lines += ["![](position_heatmap.png)", ""]

    # ---- P5: counterfactual anchor (who wins: the overridden JPM point or the instruction?)
    cf = by_probe.get("anchor_cf", [])
    if cf:
        lines += ["## P5 counterfactual anchor", "", "| guidance λ | n | follows anchor | follows language |", "|---|---|---|---|"]
        by_l = defaultdict(list)
        for r in cf:
            by_l[r.get("guidance_lambda", "")].append(r)
        for lam, rs in sorted(by_l.items()):
            fa = sum(r["outcome"].strip() == "wrong_object" and r["grabbed"].strip() == r["anchor_override"].strip() for r in rs)
            fl = sum(r["outcome"].strip() == "success" for r in rs)
            lines.append(f"| {lam} | {len(rs)} | {fa/len(rs):.2f} | {fl/len(rs):.2f} |")
        lines.append("")

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
