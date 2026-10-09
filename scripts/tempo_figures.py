"""Figures + markdown tables for TempoBench.

    python scripts/tempo_figures.py --res results_tempo --out figures
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
from tempobench import feed as F  # noqa: E402
from tempobench import world as W  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAGCOL = {0.08: "#2a78d6", 0.13: "#eb6834", 0.18: "#1baf7a"}          # validated all-pairs
KNOBCOL = {"uniform": "#2a78d6", "uniform+expire": "#eb6834", "adaptive": "#1baf7a", "arm_shift": "#eda100"}
KNOBNAME = {"uniform": "uniform speed-up", "uniform+expire": "speed-up + expire old plans",
            "adaptive": "adaptive (SAIL-like)", "arm_shift": "arm-only time shift"}
FAM = {"base": "#2a78d6", "ft": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def load(res):
    e1 = json.load(open(os.path.join(res, "e1_formula.json"))) if os.path.exists(os.path.join(res, "e1_formula.json")) else []
    e2 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "e2_speed_s*.json")))]
    e3 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "e3_models_s*.json")))]
    cfg = json.load(open(os.path.join(res, "config.json")))
    return e1, e2, e3, cfg


def fig_formula(e1, out):
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for life, mk, lab in [(1.2, "o", "blend life 1.2 s (served)"), (0.6, "s", "blend life 0.6 s")]:
        pts = [(r["predicted"], r["measured"]) for r in e1 if r["blend_life"] == life]
        if not pts:
            continue
        p, m = np.array(pts).T
        ax.scatter(p, m, marker=mk, s=26, color="#2a78d6" if life == 1.2 else "#eb6834", label=lab, zorder=3,
                   edgecolor=SURF, linewidth=0.8)
    lo, hi = 0.4, 1.5
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=2)
    ax.text(1.38, 1.30, "y = x", fontsize=7, color=INK2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted executed / planned speed")
    ax.set_ylabel("measured (long cruise, oracle plans)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "formula_check.png"), dpi=170)
    plt.close(fig)


def _agg(e2, ds, variant, key):
    knobs = sorted({r["knob"] for r in e2[0] if r["delta_sim"] == ds and r["variant"] == variant})
    M = np.array([[next(r[key] for r in run if r["delta_sim"] == ds and r["variant"] == variant and r["knob"] == k)
                   for k in knobs] for run in e2])
    return np.array(knobs), M.mean(0), M.std(0)


def fig_speed(e2, cfg, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ds in sorted(LAGCOL):
        for ax, key in zip(axes, ["score", "subtask_success", "tempo_ratio_abs"]):
            k, mu, sd = _agg(e2, ds, "uniform", key)
            ax.plot(k, mu, color=LAGCOL[ds], lw=2, marker="o", ms=4, label=f"robot lag {ds:.2f} s")
            ax.fill_between(k, mu - sd, mu + sd, color=LAGCOL[ds], alpha=0.15, lw=0)
    s_pred = cfg["s_star_pred"]
    for ds in sorted(LAGCOL):
        axes[2].axvline(s_pred[str(ds)], color=LAGCOL[ds], ls=":", lw=1.2)
    axes[2].axhline(1.0, color=INK2, lw=1, ls="--")
    for ax in axes:
        ax.axvspan(1.425, 2.45, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.set_xlabel("playback speed s (row_dt = 1/30 s ÷ s)")
    yl = axes[1].get_ylim()
    axes[1].text(1.46, yl[0] + 0.03 * (yl[1] - yl[0]), "s > 1.42: oldest plans\nfreeze in the blend", fontsize=6.5, color=INK2)
    axes[0].set_ylabel("score per episode (max 2)")
    axes[1].set_ylabel("subtasks completed")
    axes[2].set_ylabel("executed / planned speed")
    axes[2].text(2.0, 0.975, "deficit closed", fontsize=6.5, color=INK2, va="top")
    axes[0].legend(fontsize=7, frameon=False)
    axes[2].set_title("dotted: formula's speed that closes the deficit", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speed_curves.png"), dpi=170)
    plt.close(fig)


def fig_knobs(e2, e2b, out):
    cats = [("none", "speed 1 (served)", "#8a8986"), ("uniform", "uniform speed-up", "#2a78d6"),
            ("uniform+expire", "speed-up + expire old plans", "#eda100"),
            ("adaptive_jitter", "adaptive, row-jitter detector", "#1baf7a"),
            ("adaptive_crossing", "adaptive, crossing detector", "#eb6834"),
            ("arm_shift", "arm-only time shift", "#4a3aa7")]
    lags = sorted(LAGCOL)
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    w = 0.13
    for ci, (key, lab, col) in enumerate(cats):
        for li, ds in enumerate(lags):
            if key == "none":
                _, mu, sd = _agg(e2, ds, "uniform", "score"); m, d, kn = mu[0], sd[0], None
            elif key.startswith("adaptive_"):
                if not e2b:
                    continue
                runs = [[r for r in run["rows"] if r["delta_sim"] == ds and r["variant"] == key] for run in e2b]
                knobs = sorted({r["knob"] for r in runs[0]})
                M = np.array([[next(r["score"] for r in run if r["knob"] == k) for k in knobs] for run in runs])
                b = int(np.argmax(M.mean(0))); m, d, kn = M.mean(0)[b], M.std(0)[b], knobs[b]
            else:
                k, mu, sd = _agg(e2, ds, key, "score"); b = int(np.argmax(mu)); m, d, kn = mu[b], sd[b], k[b]
            x = li + (ci - (len(cats) - 1) / 2) * w
            ax.bar(x, m, w - 0.015, color=col, label=lab if li == 0 else None, zorder=2)
            if d > 0:
                ax.errorbar(x, m, d, color=INK, lw=0.8, capsize=1.5, zorder=3)
            ax.text(x, m + max(d, 0) + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=5.8, color=INK)
            if kn is not None:
                ax.text(x, 0.02, f"{kn:g}", ha="center", va="bottom", fontsize=5.2, color="white", rotation=90)
    ax.set_xticks(range(len(lags)), [f"robot lag {d:.2f} s (demo 0.08 s)" for d in lags])
    ax.set_ylabel("best score per episode (max 2)")
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="x", visible=False)
    ax.text(1.0, -0.17, "white numbers: the knob value that gave the best score (speed, or shift in s)",
            transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "knobs.png"), dpi=170)
    plt.close(fig)


def fig_models(e3, out):
    rows = [r for run in e3 for r in run]
    y = np.array([r["cl_score"] for r in rows])
    fam = ["ft" if r["model"].startswith("ft") else "base" for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for ax, key, lab in [(axes[0], "err_teleop_val", "open-loop error vs held-out teleop demos"),
                         (axes[1], "planned_tempo", "planned tempo (offline): plan / demo speed")]:
        x = np.array([r[key] for r in rows])
        for xi, yi, f in zip(x, y, fam):
            ax.scatter(xi, yi, s=26, color=FAM[f], edgecolor=SURF, linewidth=0.8, zorder=3,
                       marker="o" if f == "base" else "D")
        ax.set_xlabel(lab, fontsize=8)
        ax.set_title(f"Spearman ρ = {spearman(x, y):+.2f}", fontsize=8.5, color=INK)
        ax.set_ylabel("closed-loop score at speed 1")
    ex = np.array([r["err_teleop_val"] for r in rows]); tp = np.array([r["planned_tempo"] for r in rows])
    sc = axes[2].scatter(ex, tp, c=y, cmap="Blues", vmin=0, vmax=max(0.7, y.max()), s=60,
                         edgecolor=INK2, linewidth=0.6, zorder=3)
    names = sorted({r["model"] for r in rows}, key=lambda m: [r["model"] for r in rows].index(m))
    for k, nm in enumerate(names):
        mx = np.mean([r["err_teleop_val"] for r in rows if r["model"] == nm])
        mt = np.mean([r["planned_tempo"] for r in rows if r["model"] == nm])
        dy = 0.014 if nm != "ft_50" else -0.03
        axes[2].text(mx, mt + dy, nm, fontsize=6.3, color=INK2, ha="center")
    cb = fig.colorbar(sc, ax=axes[2], fraction=0.05, pad=0.02)
    cb.set_label("closed-loop score", fontsize=7.5)
    axes[2].set_xlabel("open-loop error vs teleop demos", fontsize=8)
    axes[2].set_ylabel("planned tempo", fontsize=8)
    axes[2].set_title("both axes are needed", fontsize=8.5, color=INK)
    h1 = axes[0].scatter([], [], color=FAM["base"], label="scripted-only (2k / 4k / 8k steps)")
    h2 = axes[0].scatter([], [], color=FAM["ft"], marker="D", label="fine-tuned on 25 / 50 / 100 % teleop")
    fig.legend(handles=[h1, h2], ncol=2, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.36, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "offline_vs_closed_loop.png"), dpi=170)
    plt.close(fig)


def fig_leaderboard(lb, out):
    if not lb:
        return
    pts = sorted((v["speed"], v["score"], k) for k, v in lb.items() if v.get("kind") == "uniform")
    other = [(k, v) for k, v in lb.items() if v.get("kind") != "uniform"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    s, sc, names = zip(*pts)
    ax.plot(s, sc, color="#2a78d6", lw=2, marker="o", ms=5, zorder=3)
    for si, ci, n in pts:
        ax.text(si, ci + 0.25, f"{ci:.2f}", ha="center", fontsize=7, color=INK)
    # variants without a speed change: one column left of the speed axis, labels spread apart
    other = sorted(other, key=lambda kv: kv[1]["score"])
    ly = []
    for k, v in other:
        y = v["score"]
        ly.append(max(y, ly[-1] + 0.32) if ly else y)
    for (k, v), yl in zip(other, ly):
        ax.scatter(0.9, v["score"], marker="D", s=28, color="#eb6834", zorder=4)
        ax.annotate(f"{k} {v['score']:.2f}", (0.9, v["score"]), xytext=(0.86, yl), textcoords="data",
                    ha="right", va="center", fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.6))
    ax.axvline(0.95, color=GRID, lw=1)
    ymin = min(v["score"] for _, v in other)
    ax.text(0.9, ymin - 0.45, "speed 1,\nother change", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.set_ylim(ymin - 1.0, max(sc) + 0.6)
    ax.axvspan(1.425, max(2.05, max(s) + 0.05), color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(1.44, max(sc) - 0.3, "s > 1.42: oldest plans\nin the blend freeze", fontsize=6.5, color=INK2, va="top")
    ax.set_xlim(0.55, max(2.05, max(s) + 0.05))
    ax.set_xticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel("playback speed of the plan")
    ax.set_ylabel("leaderboard score (of 88)")
    ax.set_title("Green Challenge leaderboard. Blue: baseline checkpoint, only playback speed changed", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leaderboard.png"), dpi=170)
    plt.close(fig)


def fig_timeseries(res, out, ds=0.13):
    """One episode, same start: hand distance to the object over time at speed 1 and 1.3,
    with the distance the policy's own plan wanted to cover (rows at demo pace)."""
    from tempobench import closed_loop as C
    from tempobench import policy as P
    ck = os.path.join(res, "ckpt", "base_s0.pkl")
    if not os.path.exists(ck):
        return
    pol = P.FlowPolicy(P.load(ck), seed=0)
    TO = W.calibrate_timeouts()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for sp, col in [(1.0, "#2a78d6"), (1.3, "#eb6834")]:
        r = C.rollout(pol, F.FeedConfig(speed=sp), n=40, seed=3, omega=W.omega_for_lag(ds), timeouts=TO,
                      record_traj=40)
        tr = r["traj"]                                       # (n, ticks, 4): hand xy, command xy
        rng = np.random.RandomState(10_000 + 3)
        w0 = W.World(40, rng, omega=W.omega_for_lag(ds), timeouts=TO)
        d = np.linalg.norm(tr[:, :, :2] - w0.obj[:, None], axis=-1)
        t = (np.arange(tr.shape[1]) + 1) * W.DT
        m = np.median(d / d[:, :1], 0)
        ax.plot(t, m, color=col, lw=2, label=f"speed {sp:g}: hand (median of 40 episodes)")
        dc = np.linalg.norm(tr[:, :, 2:] - w0.obj[:, None], axis=-1)
        ax.plot(t, np.median(dc / d[:, :1], 0), color=col, lw=1, ls="--", label=f"speed {sp:g}: set-point after feed")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("time, s")
    ax.set_ylabel("distance to object / initial")
    ax.set_title(f"pick phase, robot lag {ds:.2f} s", fontsize=8.5, color=INK)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "timeseries.png"), dpi=170)
    plt.close(fig)


def tables(e1, e2, e3, cfg, out):
    L = []
    if e1:
        err = np.array([r["measured"] - r["predicted"] for r in e1 if r["blend_life"] == 1.2])
        err6 = np.array([r["measured"] - r["predicted"] for r in e1 if r["blend_life"] == 0.6])
        L += [f"E1: measured − predicted tempo ratio, blend 1.2 s: mean {err.mean():+.3f}, |max| {np.abs(err).max():.3f}; "
              f"blend 0.6 s: mean {err6.mean():+.3f}, |max| {np.abs(err6).max():.3f} (n={len(err)}+{len(err6)})", ""]
    if e2:
        L += [f"E2 (seeds: {len(e2)}), learned policy, score per episode (max 2) / subtasks completed:", "",
              "| robot lag | speed 1 | best uniform speed | score there | best expire | best adaptive | best arm shift | formula speed |",
              "|---|---|---|---|---|---|---|---|"]
        for ds in sorted(LAGCOL):
            k, mu, sd = _agg(e2, ds, "uniform", "score")
            _, su, _ = _agg(e2, ds, "uniform", "subtask_success")
            b = int(np.argmax(mu))
            cells = [f"{mu[0]:.2f} / {su[0]:.2f}", f"{k[b]:.2f}", f"{mu[b]:.2f} / {su[b]:.2f}"]
            for var in ["uniform+expire", "adaptive", "arm_shift"]:
                kk, m2, _ = _agg(e2, ds, var, "score")
                j = int(np.argmax(m2))
                cells.append(f"{m2[j]:.2f} @ {kk[j]:g}")
            cells.append(f"{cfg['s_star_pred'][str(ds)]:.2f}")
            L.append(f"| {ds:.2f} s | " + " | ".join(cells) + " |")
        L.append("")
        L += ["Measured speed at which executed/planned crosses 1 (linear interpolation of the mean curve) vs formula:", ""]
        for ds in sorted(LAGCOL):
            k, tr, _ = _agg(e2, ds, "uniform", "tempo_ratio_abs")
            j = int(np.argmax(tr >= 1.0))
            cross = k[j - 1] + (1.0 - tr[j - 1]) * (k[j] - k[j - 1]) / (tr[j] - tr[j - 1]) if j > 0 else float("nan")
            L.append(f"- lag {ds:.2f} s: tempo at speed 1 = {tr[0]:.3f}; crosses 1 at s = {cross:.2f}; formula {cfg['s_star_pred'][str(ds)]:.2f}")
        L.append("")
    if e3:
        names = [r["model"] for r in e3[0]]
        L += [f"E3 (seeds: {len(e3)}), mean over seeds:", "",
              "| model | open-loop error, teleop val | open-loop error, scripted val | planned tempo | "
              "closed-loop success @1 | closed-loop score @1 | calibrated speed | score @ calibrated | best score in sweep (speed) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for nm in names:
            rr = [next(r for r in run if r["model"] == nm) for run in e3]
            f = lambda k: np.mean([r[k] for r in rr])  # noqa: E731
            sw = np.mean([[x[1] for x in r["sweep"]] for r in rr], 0)
            sp = [x[0] for x in rr[0]["sweep"]]
            j = int(np.argmax(sw))
            L.append(f"| {nm} | {f('err_teleop_val'):.4f} | {f('err_scripted_val'):.4f} | {f('planned_tempo'):.2f} | "
                     f"{f('cl_subtask_success'):.2f} | {f('cl_score'):.2f} | {f('s_cal'):.2f} | {f('cal_score'):.2f} | "
                     f"{sw[j]:.2f} ({sp[j]:g}) |")
        L += ["", "Share of the sweep gain (best − speed 1) recovered by the calibrated speed:"]
        for nm in names:
            rr = [next(r for r in run if r["model"] == nm) for run in e3]
            sw = np.mean([[x[1] for x in r["sweep"]] for r in rr], 0)
            s1, sc, sb = np.mean([r["cl_score"] for r in rr]), np.mean([r["cal_score"] for r in rr]), sw.max()
            L.append(f"- {nm}: {(sc - s1) / max(sb - s1, 1e-9) * 100:.0f} %")
        rows = [r for run in e3 for r in run]
        y = [r["cl_score"] for r in rows]
        L += ["", "Spearman ρ with closed-loop score @1 over all checkpoints × seeds: " + ", ".join(
            f"{k} {spearman([r[k] for r in rows], y):+.2f}" for k in ["err_teleop_val", "err_scripted_val", "planned_tempo"])]
    open(os.path.join(out, "results_tempo.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def extra_tables(res, out):
    L = []
    p = os.path.join(res, "e1b_blend.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        L += ["E1b: blend on/off and blend life on the task (seed 0; score / subtasks / executed÷planned speed):", "",
              "| planner | robot lag | no blend | life 0.4 s | life 0.8 s | life 1.2 s (served) |", "|---|---|---|---|---|---|"]
        for pl in ("oracle", "learned"):
            for ds in sorted(LAGCOL):
                rr = [r for r in rows if r["planner"] == pl and r["delta_sim"] == ds]
                order = sorted(rr, key=lambda r: (r["blend"], r["life"]))
                L.append(f"| {pl} | {ds:.2f} s | " + " | ".join(
                    f"{r['score']:.2f} / {r['subtask_success']:.2f} / {r['tempo_ratio_abs']:.2f}" for r in order) + " |")
        L.append("")
    runs = [json.load(open(q)) for q in sorted(glob.glob(os.path.join(res, "e2b_adaptive_s*.json")))]
    if runs:
        L += [f"E2b ({len(runs)} seeds): rows flagged as a gripper event", "",
              "| chunks | row-jitter detector | crossing detector |", "|---|---|---|"]
        for k in ("demo_chunks", "policy_plans"):
            L.append(f"| {k} | {np.mean([r['detector'][k]['rows_flagged_jitter'] for r in runs]):.3f} | "
                     f"{np.mean([r['detector'][k]['rows_flagged_crossing'] for r in runs]):.3f} |")
        L += ["", "| robot lag | detector | 1.5 | 2.0 | 2.5 | 3.0 | rows kept at 1x |", "|---|---|---|---|---|---|---|"]
        for ds in sorted(LAGCOL):
            for det in ("jitter", "crossing"):
                xs = [[x for x in r["rows"] if x["delta_sim"] == ds and x["variant"] == "adaptive_" + det] for r in runs]
                M = np.array([[x["score"] for x in X] for X in xs]); S = np.array([[x["subtask_success"] for x in X] for X in xs])
                sl = np.mean([[x["slow_rows_frac"] for x in X] for X in xs])
                L.append(f"| {ds:.2f} s | {det} | " + " | ".join(f"{m:.2f} / {su:.2f}" for m, su in zip(M.mean(0), S.mean(0)))
                         + f" | {sl:.2f} |")
        L += ["", "| robot lag | speed | clamp on: executed÷planned / score | clamp off |", "|---|---|---|---|"]
        for ds in sorted(LAGCOL):
            for sp in (1.0, 1.3):
                on = [next(x for x in r["rows"] if x["delta_sim"] == ds and x["variant"] == "clamp_on" and x["knob"] == sp) for r in runs]
                off = [next(x for x in r["rows"] if x["delta_sim"] == ds and x["variant"] == "clamp_off" and x["knob"] == sp) for r in runs]
                f = lambda Lx, k: np.mean([x[k] for x in Lx])  # noqa: E731
                L.append(f"| {ds:.2f} s | {sp} | {f(on, 'tempo_ratio_abs'):.3f} / {f(on, 'score'):.2f} | "
                         f"{f(off, 'tempo_ratio_abs'):.3f} / {f(off, 'score'):.2f} |")
    if L:
        with open(os.path.join(out, "results_tempo.md"), "a") as fh:
            fh.write("\n" + "\n".join(L) + "\n")
        print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_tempo")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--leaderboard", default="data/leaderboard.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    e1, e2, e3, cfg = load(a.res)
    e2b = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.res, "e2b_adaptive_s*.json")))]
    if e1:
        fig_formula(e1, a.out)
    if e2:
        fig_speed(e2, cfg, a.out)
        fig_knobs(e2, e2b, a.out)
    if e3:
        fig_models(e3, a.out)
    lb = json.load(open(a.leaderboard)) if os.path.exists(a.leaderboard) else {}
    fig_leaderboard(lb, a.out)
    fig_timeseries(a.res, a.out)
    tables(e1, e2, e3, cfg, a.out)
    extra_tables(a.res, a.out)


if __name__ == "__main__":
    main()
