"""Aggregate results/seed_*.json into figures and a markdown table.

    python scripts/make_figures.py --res results --out figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anchorbench import env as E  # noqa: E402

# validated with the dataviz palette checker (adjacent pairs, light surface)
MAIN = ["bc", "bc_cotrain", "guide", "inject", "anchor_frame"]
MAIN_COL = {"bc": "#2a78d6", "bc_cotrain": "#1baf7a", "guide": "#eda100", "inject": "#4a3aa7",
            "anchor_frame": "#eb6834"}
NAME = {"bc": "BC (FiLM-CNN)", "bc_cotrain": "BC + pointing co-train", "guide": "Guidance (JPM-like)",
        "inject": "Point injection", "anchor_frame": "Anchor frame (ours)",
        "af_nocrop": "AF w/o local crop", "af_global": "AF + global image"}
SPLIT_NAME = {"id": "in-distribution", "pos_ood": "new positions", "combo_ood": "new colour×shape",
              "swap": "swapped places", "bg": "new table colour"}
OUT_COL = {"success": "#2a78d6", "wrong_object": "#eb6834", "miss": "#1baf7a",
           "collision": "#eda100", "timeout": "#e87ba4"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def load(res):
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "seed_*.json")))]
    if not runs:
        raise SystemExit(f"no results in {res}")
    return runs


def stat(runs, f):
    v = np.array([f(r) for r in runs], float)
    return v.mean(), v.std(), v


def fig_success(runs, out):
    splits = ["id", "pos_ood", "combo_ood", "swap", "bg"]
    methods = [m for m in MAIN if m in runs[0]["splits"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, s in enumerate(splits):
            mu, sd, v = stat(runs, lambda r: r["splits"][m][s]["success"])
            x = i + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, mu, w - 0.02, color=MAIN_COL[m], label=NAME[m] if i == 0 else None, zorder=2)
            if len(v) > 1:
                ax.errorbar(x, mu, sd, color=INK, lw=1, capsize=2, zorder=3)
            if m in ("bc", "anchor_frame"):
                ax.text(x, mu + max(sd, 0) + 0.02, f"{mu:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(splits)), [SPLIT_NAME[s] for s in splits])
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_by_split.png"), dpi=170)
    plt.close(fig)


def fig_failures(runs, out):
    methods = [m for m in MAIN if m in runs[0]["splits"]]
    cats = ["success", "wrong_object", "miss", "collision", "timeout"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
    for ax, s in zip(axes, ["pos_ood", "swap"]):
        bottom = np.zeros(len(methods))
        for c in cats:
            vals = np.array([stat(runs, lambda r: r["splits"][m][s][c])[0] for m in methods])
            ax.bar(range(len(methods)), vals, 0.7, bottom=bottom, color=OUT_COL[c], label=c.replace("_", " "),
                   edgecolor=SURF, linewidth=1.5, zorder=2)
            for i, (b, v) in enumerate(zip(bottom, vals)):
                if v > 0.08:
                    ax.text(i, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white")
            bottom += vals
        ax.set_xticks(range(len(methods)), [NAME[m].replace(" (", "\n(").replace(" + ", "\n+ ") for m in methods],
                      fontsize=6.8)
        ax.set_title(SPLIT_NAME[s], fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("fraction of episodes")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "failure_modes.png"), dpi=170)
    plt.close(fig)


def fig_probes(runs, out):
    """Left: memorisation probe (target removed). Right: counterfactual anchor."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9))
    methods = [m for m in MAIN if m in runs[0]["splits"]]
    ax = axes[0]
    for i, m in enumerate(methods):
        mu, sd, _ = stat(runs, lambda r: r["splits"][m]["empty"]["air_at_usual_place"])
        ax.bar(i, mu, 0.7, color=MAIN_COL[m], zorder=2)
        ax.errorbar(i, mu, sd, color=INK, lw=1, capsize=2)
        ax.text(i, mu + sd + 0.02, f"{mu:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(methods)), [NAME[m].replace(" (", "\n(").replace(" + ", "\n+ ") for m in methods], fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("target removed: closes the gripper\nat the target's usual place", fontsize=8.5, color=INK)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    am = [m for m in ["guide", "inject", "anchor_frame"] if m in runs[0]["probes"]]
    for i, m in enumerate(am):
        fa, sa, _ = stat(runs, lambda r: r["probes"][m]["counterfactual"]["follows_anchor"])
        fl, sl, _ = stat(runs, lambda r: r["probes"][m]["counterfactual"]["follows_language"])
        ax.bar(i - 0.18, fa, 0.34, color=MAIN_COL[m], zorder=2)
        ax.bar(i + 0.18, fl, 0.34, color=MAIN_COL[m], alpha=0.35, hatch="////", edgecolor=MAIN_COL[m], zorder=2)
        ax.text(i - 0.18, fa + 0.02, f"{fa:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + 0.18, fl + 0.02, f"{fl:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(am)), [NAME[m].replace(" (", "\n(") for m in am], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("anchor moved to a distractor, instruction unchanged\nsolid: grabs the anchored object · hatched: grabs the named one",
                 fontsize=8, color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "causal_probes.png"), dpi=170)
    plt.close(fig)


def fig_noise(runs, out):
    sig = [0.0, 0.02, 0.04, 0.06, 0.08]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    lines = [("anchor_frame", "#eb6834", "-", "o"), ("inject", "#4a3aa7", "-", "s"), ("af_global", "#2a78d6", "-", "^"),
             ("af_nocrop", "#8a8986", "--", "v"), ("guide", "#8a8986", ":", "d")]
    for m, c, ls, mk in lines:
        if m not in runs[0]["probes"]:
            continue
        mus = []
        for s in sig:
            key = "oracle_pos_ood" if s == 0 else f"noise_{s}"
            mus.append(stat(runs, lambda r: r["probes"][m][key]["success"])[0])
        ax.plot(np.array(sig) * 100, mus, color=c, ls=ls, marker=mk, ms=5, lw=2)
        ax.text(sig[-1] * 100 + 0.25, mus[-1], NAME[m], fontsize=7, color=INK2, va="center")
    ax.axvline(E.R * 100, color=GRID, lw=1.5, zorder=0)
    ax.text(E.R * 100, 1.02, "object radius", fontsize=6.5, color=INK2, ha="center")
    ax.set_xlabel("systematic pointing error σ, % of workspace")
    ax.set_ylabel("success, new positions")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.3, 12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pointing_error.png"), dpi=170)
    plt.close(fig)


def fig_ablation(runs, out):
    splits = ["id", "pos_ood", "combo_ood", "swap", "bg"]
    methods = [m for m in ["anchor_frame", "af_global", "af_nocrop"] if m in runs[0]["splits"]]
    col = {"anchor_frame": "#eb6834", "af_global": "#2a78d6", "af_nocrop": "#4a3aa7"}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        for i, s in enumerate(splits):
            mu, sd, _ = stat(runs, lambda r: r["splits"][m][s]["success"])
            x = i + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, mu, w - 0.03, color=col[m], label=NAME[m] if i == 0 else None, zorder=2)
            ax.errorbar(x, mu, sd, color=INK, lw=1, capsize=2)
    ax.set_xticks(range(len(splits)), [SPLIT_NAME[s] for s in splits], fontsize=8)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation.png"), dpi=170)
    plt.close(fig)


def fig_data(runs, out):
    """Where the target object is in the demonstrations vs in the 'new positions' test."""
    sys.path.insert(0, os.path.dirname(__file__))
    from run_experiment import eval_scenes
    tr = np.array(runs[0]["data"]["train_target_xy"])
    te = np.array([s.pos[0] for s in eval_scenes("pos_ood", 200)])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, pts, title in [(axes[0], tr, "robot demos: target positions"), (axes[1], te, "test 'new positions'")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color="#2a78d6", alpha=0.55, lw=0, zorder=2)
        ax.scatter(E.ANCHORS[:, 0], E.ANCHORS[:, 1], s=40, marker="+", color=INK, lw=1.2, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    axes[0].text(0.02, 0.03, "+ = usual place of each of the 12 objects", fontsize=6.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "data_layout.png"), dpi=170)
    plt.close(fig)


def fig_env(out):
    sys.path.insert(0, os.path.dirname(__file__))
    from run_experiment import eval_scenes
    splits = ["id", "pos_ood", "combo_ood", "swap", "bg", "empty"]
    fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.2))
    for ax, s in zip(axes, splits):
        sc = eval_scenes(s, 3)[1]
        ax.imshow(E.render_scenes([sc])[0], extent=[0, 1, 0, 1])
        ax.scatter(*E.ANCHORS[sc.target_id], marker="x", color="white", s=28, lw=1.2)
        ax.set_title(f"{s}\n\"{E.ident_name(sc.target_id)}\"", fontsize=7.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("evaluation splits (× = where the named object usually lies in the robot demos; white ring = gripper)",
                 fontsize=7.5, color=INK2, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, "env_splits.png"), dpi=170)
    plt.close(fig)


def fig_traj(res, out, seed=0, n=4):
    p = os.path.join(res, f"traj_seed_{seed}.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    sc = z["scene_swap"]
    K = 3
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.4))
    for i, ax in enumerate(np.atleast_1d(axes)):
        row = sc[i]
        pos = row[:2 * K].reshape(K, 2); th = row[2 * K:3 * K]; ids = row[3 * K:4 * K].astype(int)
        tid = int(row[4 * K]); present = row[4 * K + 1:5 * K + 1].astype(bool)
        ee0 = row[5 * K + 1:5 * K + 3]; bg = row[5 * K + 3:5 * K + 6]
        img = E.render_batch(pos[None], ids[None], th[None], present[None], ee0[None], bg[None])[0]
        ax.imshow(img, extent=[0, 1, 0, 1])
        for m, c in [("bc", "#86b6ef"), ("anchor_frame", "#eb6834")]:
            key = f"{m}/swap"
            if key in z:
                t = z[key][i]
                keep = np.r_[True, np.linalg.norm(np.diff(t, axis=0), axis=1) > 1e-6]
                t = t[keep]
                ax.plot(t[:, 0], t[:, 1], color=c, lw=1.8, label=NAME[m])
                ax.scatter(t[-1, 0], t[-1, 1], color=c, s=14, zorder=4)
        ua = E.ANCHORS[tid]
        ax.scatter(*ua, marker="x", color="white", s=30, lw=1.2)
        ax.set_title(f"\"pick the {E.ident_name(tid)}\"", fontsize=7.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    np.atleast_1d(axes)[0].legend(fontsize=6, loc="lower left", framealpha=0.8)
    fig.suptitle("swapped places: × marks where the named object usually lies in the demos", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectories_swap.png"), dpi=170)
    plt.close(fig)


def table(runs, out):
    splits = ["id", "pos_ood", "combo_ood", "swap", "bg"]
    methods = [m for m in ["bc", "bc_cotrain", "guide", "inject", "anchor_frame", "af_global", "af_nocrop"]
               if m in runs[0]["splits"]]
    L = [f"Seeds: {len(runs)} · episodes per split per seed: {runs[0]['splits']['bc']['id']['n']}", "",
         "| method | " + " | ".join(SPLIT_NAME[s] for s in splits) + " | air-grasp, target removed | wrong object, swapped |",
         "|---|" + "---|" * (len(splits) + 2)]
    for m in methods:
        cells = []
        for s in splits:
            mu, sd, _ = stat(runs, lambda r: r["splits"][m][s]["success"])
            cells.append(f"{mu:.2f} ± {sd:.2f}")
        mu, sd, _ = stat(runs, lambda r: r["splits"][m]["empty"]["air_at_usual_place"])
        cells.append(f"{mu:.2f} ± {sd:.2f}")
        mu, sd, _ = stat(runs, lambda r: r["splits"][m]["swap"]["wrong_object"])
        cells.append(f"{mu:.2f} ± {sd:.2f}")
        L.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    L += ["", "| method (probe) | pointing acc., new positions | success w/ oracle point, new positions | "
          "success w/ oracle point, new colour×shape | follows moved anchor | follows language |", "|---|---|---|---|---|---|"]
    for m in [m for m in ["guide", "inject", "anchor_frame", "af_global", "af_nocrop"] if m in runs[0]["probes"]]:
        ga = stat(runs, lambda r: r["splits"][m]["pos_ood"]["grounding_acc"])[0]
        op = stat(runs, lambda r: r["probes"][m]["oracle_pos_ood"]["success"])
        oc = stat(runs, lambda r: r["probes"][m]["oracle_combo_ood"]["success"])
        fa = stat(runs, lambda r: r["probes"][m]["counterfactual"]["follows_anchor"])
        fl = stat(runs, lambda r: r["probes"][m]["counterfactual"]["follows_language"])
        L.append(f"| {NAME[m]} | {ga:.2f} | {op[0]:.2f} ± {op[1]:.2f} | {oc[0]:.2f} ± {oc[1]:.2f} | "
                 f"{fa[0]:.2f} ± {fa[1]:.2f} | {fl[0]:.2f} ± {fl[1]:.2f} |")
    L += ["", "| method | σ=0 | σ=0.02 | σ=0.04 | σ=0.06 | σ=0.08 |", "|---|---|---|---|---|---|"]
    for m in [m for m in ["inject", "anchor_frame", "af_global", "af_nocrop", "guide"] if m in runs[0]["probes"]]:
        row = [stat(runs, lambda r: r["probes"][m]["oracle_pos_ood"]["success"])[0]]
        row += [stat(runs, lambda r, s=s: r["probes"][m][f"noise_{s}"]["success"])[0] for s in (0.02, 0.04, 0.06, 0.08)]
        L.append(f"| {NAME[m]} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    open(os.path.join(out, "results_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_env(a.out)
    runs = load(a.res)
    fig_data(runs, a.out)
    fig_success(runs, a.out)
    fig_failures(runs, a.out)
    fig_probes(runs, a.out)
    fig_noise(runs, a.out)
    fig_ablation(runs, a.out)
    fig_traj(a.res, a.out)
    table(runs, a.out)


if __name__ == "__main__":
    main()
