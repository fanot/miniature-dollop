"""Vectorised replica of the Green Challenge action feed (organisers' action_feed.py).

Mechanisms, in the order they run (same as the original):
 1. Scheduling: row j of a plan observed at t_obs is due at t_obs + lead + (j+1)*row_dt; the
    command at time t samples each plan at fractional row (t - t_obs - lead)/row_dt - 1.
    A fresh plan is pending (ignored) until its row 0 is due. Speed-up s = (1/30)/row_dt.
 2. Blending: the set-point is the exp(-age/tau)-weighted mean of every plan younger than
    blend_life (the newest active plan always takes part). A plan older than its horizon
    *freezes* on its last row (original behaviour); `expire=True` drops it instead (the fix the
    green_challenge patch applies only to adaptive speed).
 3. Clamping: velocity and acceleration of the xy set-point limited against the previous command.

Anchoring: plans arrive as deltas to the hand position observed at t_obs (Green's delta
actions). `anchor="state"` adds them to that observation (what the contest server does).
`anchor="lagfree"` is the fix studied in the README: it adds them to
    command(t_obs) - delta_demo * velocity(t_obs),
i.e. to where the hand *would* be if it tracked like the demonstration robot.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import world as W


@dataclasses.dataclass
class FeedConfig:
    speed: float = 1.0          # row_dt = FRAME / speed
    lead: float = 0.030
    period: float = 0.080
    blend: bool = True
    blend_life: float = 1.20
    blend_tau: float = 1.00
    expire: bool = False        # drop plans past their last row instead of freezing them
    vlim: float = 1.5           # xy speed limit (units/s)
    alim: float = 6.0           # xy acceleration limit (units/s^2)
    anchor: str = "state"       # "state" | "lagfree"
    delta_demo: float = W.lag_of(W.OMEGA_DEMO)
    adaptive_speed: float = 1.0  # SAIL-style: free-motion rows faster, rows near gripper events at 1x
    adaptive_window: int = 10
    adaptive_detector: str = "jitter"   # "jitter": any row-to-row gripper change > 0.02 (logic of my green_challenge patch)
                                        # "crossing": rows where the plan's gripper crosses 0.5
    arm_shift: float = 0.0       # sample the arm (xy) rows this many seconds ahead; gripper untouched

    @property
    def row_dt(self):
        return W.FRAME / self.speed


class Feed:
    def __init__(self, n, cfg: FeedConfig, H=W.H, kmax=24):
        self.cfg, self.n, self.H, self.K = cfg, n, H, kmax
        self.rows = np.zeros((n, kmax, H, 3))
        self.times = np.zeros((n, kmax, H))   # due time of each row relative to t_obs
        self.tobs = np.full((n, kmax), -1e9)
        self.slot = 0
        self.first_tobs = None
        self.prev = None
        self.prev_v = np.zeros((n, 2))
        self.n_live_max = 0
        self.clamped = 0
        self.ticks = 0

    def reset(self, p, g):
        self.prev = np.concatenate([p, g[:, None]], 1).copy()

    def _row_times(self, rows):
        cfg = self.cfg
        base = cfg.lead + cfg.row_dt * (np.arange(self.H) + 1.0)
        if cfg.adaptive_speed <= 1.0 + 1e-9:
            return np.broadcast_to(base, (rows.shape[0], self.H)).copy()
        fing = rows[:, :, 2]
        if cfg.adaptive_detector == "crossing":
            ev = np.diff((fing > 0.5).astype(int), axis=1) != 0
        else:
            ev = np.abs(np.diff(fing, axis=1)) > 0.02
        self.slow_rows_frac = getattr(self, "slow_rows_frac", [])
        slow = np.zeros_like(fing, bool)
        for j in range(self.H - 1):
            lo, hi = max(0, j + 1 - cfg.adaptive_window), min(self.H, j + 2 + cfg.adaptive_window)
            slow[:, lo:hi] |= ev[:, j:j + 1]
        self.slow_rows_frac.append(float(slow.mean()))
        step = np.where(slow[:, 1:], cfg.row_dt, cfg.row_dt / cfg.adaptive_speed)
        t = np.empty_like(fing)
        t[:, 0] = cfg.lead + cfg.row_dt
        t[:, 1:] = t[:, :1] + np.cumsum(step, 1)
        return t

    def push(self, t_obs, delta_rows, p_obs, v_obs):
        """delta_rows (n,H,3): xy deltas + absolute gripper."""
        cfg = self.cfg
        if cfg.anchor == "state":
            anchor = p_obs
        elif cfg.anchor == "lagfree":
            anchor = self.prev[:, :2] - cfg.delta_demo * v_obs
        else:
            raise ValueError(cfg.anchor)
        rows = delta_rows.astype(np.float64).copy()
        rows[:, :, :2] += anchor[:, None, :]
        k = self.slot % self.K
        if self.first_tobs is None:
            self.first_tobs = t_obs        # the first chunk of an episode takes effect at once
        self.rows[:, k] = rows
        self.times[:, k] = self._row_times(rows)
        self.tobs[:, k] = t_obs
        self.slot += 1

    def _sample(self, t):
        val, age, j = self._sample_at(t)
        if self.cfg.arm_shift != 0.0:
            val_s, _, _ = self._sample_at(t + self.cfg.arm_shift)
            val = np.concatenate([val_s[..., :2], val[..., 2:]], -1)
        return val, age, j

    def _sample_at(self, t):
        """Each plan's forecast for time t: (n,K,3), plus age (n,K)."""
        age = t - self.tobs                                   # (n,K)
        T = self.times                                        # (n,K,H)
        j = (T <= age[..., None]).sum(-1) - 1                 # last row already due
        j0 = np.clip(j, 0, self.H - 1)
        j1 = np.clip(j + 1, 0, self.H - 1)
        t0 = np.take_along_axis(T, j0[..., None], -1)[..., 0]
        t1 = np.take_along_axis(T, j1[..., None], -1)[..., 0]
        frac = np.where(j1 > j0, np.clip((age - t0) / np.maximum(t1 - t0, 1e-9), 0, 1), 0.0)
        frac = np.where(j < 0, 0.0, frac)
        r0 = np.take_along_axis(self.rows, j0[..., None, None].repeat(3, -1), 2)[:, :, 0]
        r1 = np.take_along_axis(self.rows, j1[..., None, None].repeat(3, -1), 2)[:, :, 0]
        val = r0 + frac[..., None] * (r1 - r0)
        return val, age, j

    def command(self, t):
        cfg = self.cfg
        val, age, j = self._sample(t)
        due = (j >= 0) | (self.tobs == self.first_tobs)       # row 0 already due
        # newest plan in force
        tob = np.where(due, self.tobs, -np.inf)
        newest = np.argmax(tob, 1)
        has = np.isfinite(tob.max(1))
        if cfg.blend:
            live = due & (age <= cfg.blend_life + 1e-9)
            if cfg.expire or cfg.adaptive_speed > 1.0 + 1e-9:
                live &= age <= self.times[..., -1] + 1e-9
            live[np.arange(self.n), newest] |= has
            wgt = np.where(live, np.exp(-age / cfg.blend_tau), 0.0)
            self.n_live_max = max(self.n_live_max, int(live.sum(1).max()))
        else:
            wgt = np.zeros_like(age)
            wgt[np.arange(self.n), newest] = has.astype(float)
        ws = wgt.sum(1, keepdims=True)
        cmd = (wgt[..., None] * val).sum(1) / np.maximum(ws, 1e-12)
        cmd = np.where(ws > 0, cmd, self.prev)
        # clamp xy against the previous command
        v = (cmd[:, :2] - self.prev[:, :2]) / W.DT
        dv = v - self.prev_v
        dvn = np.linalg.norm(dv, axis=1, keepdims=True)
        lim_a = cfg.alim * W.DT
        dv = dv * np.minimum(1.0, lim_a / np.maximum(dvn, 1e-12))
        v2 = self.prev_v + dv
        vn = np.linalg.norm(v2, axis=1, keepdims=True)
        v2 = v2 * np.minimum(1.0, cfg.vlim / np.maximum(vn, 1e-12))
        self.clamped += int((np.abs(v2 - v).max(1) > 1e-6).sum())
        self.ticks += self.n
        out = cmd.copy()
        out[:, :2] = self.prev[:, :2] + v2 * W.DT
        self.prev_v = v2
        self.prev = out
        return out


def mean_plan_age(cfg: FeedConfig):
    """Weighted mean age of the plans that build the set-point (steady state), seconds."""
    first = cfg.lead + cfg.row_dt
    if not cfg.blend:
        ages = first + np.linspace(0, cfg.period, 200, endpoint=False)
        return float(ages.mean())
    ph = np.linspace(0, cfg.period, 200, endpoint=False)   # average over the phase within a period
    out = []
    for f in ph:
        ages = first + f + cfg.period * np.arange(64)
        horizon = cfg.lead + W.H * cfg.row_dt
        ok = ages <= cfg.blend_life + 1e-9
        ok[0] = True
        a = ages[ok]
        w = np.exp(-a / cfg.blend_tau)
        if not cfg.expire:
            a_eff = np.minimum(a, horizon)       # frozen plans stop advancing at the horizon
        else:
            keep = a <= horizon
            keep[0] = True
            a, a_eff, w = a[keep], a[keep], w[keep]
        out.append((a, a_eff, w))
    return out


def predicted_tempo_ratio(cfg: FeedConfig, delta_sim, delta_demo=None):
    """Steady-state executed/planned speed for a plan that cruises at speed v (README, sec. 3):

        command(t) = sum_k w_k [x(t-a_k) + v*delta_demo + s*v*(a_eff_k - lead)],  x(t) = command(t-delta_sim)

    =>  u/v = (delta_demo + s * (E_w[a_eff] - lead)) / (E_w[a] + delta_sim)
    """
    dd = cfg.delta_demo if delta_demo is None else delta_demo
    s = cfg.speed
    if not cfg.blend:
        abar = mean_plan_age(cfg)
        return (dd + s * (abar - cfg.lead)) / (abar + delta_sim)
    num, den = [], []
    for a, a_eff, w in mean_plan_age(cfg):
        ab = (w * a).sum() / w.sum()
        ae = (w * a_eff).sum() / w.sum()
        num.append(dd + s * (ae - cfg.lead)); den.append(ab + delta_sim)
    return float(np.mean(num) / np.mean(den))


def predicted_best_speed(cfg: FeedConfig, delta_sim, delta_demo=None, grid=np.linspace(0.8, 3.0, 221)):
    r = np.array([predicted_tempo_ratio(dataclasses.replace(cfg, speed=s), delta_sim, delta_demo) for s in grid])
    return float(grid[np.argmin(np.abs(r - 1.0))])
