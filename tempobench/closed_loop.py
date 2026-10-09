"""Closed-loop episodes: planner -> feed -> lagging robot, with the contest's scoring."""
from __future__ import annotations

import numpy as np

from . import world as W
from .feed import Feed, FeedConfig


def rollout(planner, cfg: FeedConfig, n=300, seed=0, omega=W.OMEGA_DEMO, timeouts=W.T_SUB,
            max_t=8.0, record_traj=0):
    rng = np.random.RandomState(10_000 + seed)
    w = W.World(n, rng, omega=omega, timeouts=timeouts)
    feed = Feed(n, cfg)
    feed.reset(w.p, w.g)
    t = 0.0
    p_prev = w.p.copy()
    # tempo bookkeeping: planned speed (plan's own motion, demo time) vs executed speed
    plan_v, exec_v, cruise = [], [], []
    traj = []
    while t < max_t - 1e-9 and w.active.any():
        v_fd = (w.p - p_prev) / cfg.period
        p_prev = w.p.copy()
        rows = planner(w)
        feed.push(t, rows, w.p, v_fd)
        pv = np.linalg.norm(rows[:, 17, :2] - rows[:, 2, :2], axis=1) / (15 * W.FRAME)
        p0 = w.p.copy()
        act0 = w.active.copy()
        for i in range(int(round(cfg.period / W.DT))):
            cmd = feed.command(t + i * W.DT)
            w.step(cmd[:, :2], cmd[:, 2], W.DT)
            if record_traj:
                traj.append(np.concatenate([w.p[:record_traj], cmd[:record_traj, :2]], 1))
        t += cfg.period
        ev = np.linalg.norm(w.p - p0, axis=1) / cfg.period
        dist = np.linalg.norm(w.target() - p0, axis=1)
        # cruising: plan wants >60 % of max speed and the target is still far away
        cr = act0 & w.active & (pv > 0.6 * W.VMAX) & (dist > 0.15) & (t > 0.4)
        plan_v.append(pv); exec_v.append(ev); cruise.append(cr)
    plan_v, exec_v, cruise = map(np.stack, (plan_v, exec_v, cruise))
    sc = w.score()
    done = ~np.isnan(w.sub_time)
    out = {
        "score": float(sc.mean()),                       # per episode, max 2
        "subtask_success": float(done.mean()),
        "pick_success": float(done[:, 0].mean()),
        "place_success": float(done[:, 1].mean()),
        "episode_success": float(done.all(1).mean()),
        "t_pick": float(np.nanmedian(w.sub_time[:, 0])) if done[:, 0].any() else float("nan"),
        "t_place": float(np.nanmedian(w.sub_time[:, 1])) if done[:, 1].any() else float("nan"),
        "missed_grasps": float(w.missed.mean()),
        "dropped": float((w.failed & (w.phase == 1) & ~w.holding & ~np.isnan(w.sub_time[:, 0])).mean()),
        "tempo_ratio": float(exec_v[cruise].sum() / max(plan_v[cruise].sum(), 1e-9) / cfg.speed)
        if cruise.any() else float("nan"),
        "tempo_ratio_abs": float(exec_v[cruise].sum() / max(plan_v[cruise].sum(), 1e-9)) if cruise.any() else float("nan"),
        "clamped_frac": feed.clamped / max(feed.ticks, 1),
        "live_plans_max": feed.n_live_max,
        "slow_rows_frac": float(np.mean(feed.slow_rows_frac)) if getattr(feed, "slow_rows_frac", None) else float("nan"),
        "n": n,
    }
    if record_traj:
        out["traj"] = np.stack(traj, 1)
    return out
