"""TempoBench world: a 2-D pick-and-place 'arm' with a lagging low-level controller.

Everything is vectorised over N parallel episodes (numpy).

Timing mirrors Green Challenge:
  * demonstrations are recorded at 30 Hz (FRAME): action[i] is the command issued at frame i;
  * deployment runs a 0.02 s control tick (DT) and asks the policy for a new plan every 0.08 s;
  * the robot itself is integrated with 5 ms substeps.

The robot follows the commanded set-point with a critically damped second-order controller
(natural frequency omega). Tracking a ramp, it lags the command by delta = 2/omega seconds; this
lag is the quantity the whole study is about. Demonstrations are recorded with omega_demo, the
deployment robot may have a different omega (a heavier arm, a softer gain, or clamping in the
feed all show up as extra lag).

Task = two subtasks, as in the contest scoring: PICK (close the gripper within R_GRASP of the
object) and PLACE (open it within R_PLACE of the target). Each has a timeout; a completed subtask
scores 1 - t/T (Green Challenge rule), a timed-out one ends the episode.
"""
from __future__ import annotations

import numpy as np

FRAME = 1.0 / 30.0      # data frame / plan row at speed 1
DT = 0.02               # control tick
SUB = 0.005             # physics substep
H = 50                  # rows per plan (Green Challenge baseline: 50 rows @ 1/30 s)

VMAX = 0.55             # expert cruise speed, workspace units / s
AMAX = 2.2              # expert acceleration limit
RHO = 0.07              # arrival radius: commanded speed ~ distance/RHO near the target
R_GRASP = 0.03
R_PLACE = 0.04
SETTLE_V = 0.06         # expert closes / opens only when the hand is this slow
TAU_G = 0.05            # gripper time constant
OMEGA_DEMO = 25.0       # demo controller -> lag 0.08 s
T_SUB = (1.75, 1.75)    # subtask timeouts (s); see calibrate_timeouts()


def lag_of(omega):
    return 2.0 / omega


def omega_for_lag(delta):
    return 2.0 / delta


class World:
    """Vectorised robot + task state."""

    def __init__(self, n, rng, omega=OMEGA_DEMO, timeouts=T_SUB):
        self.n = n
        self.omega = float(omega)
        self.timeouts = np.asarray(timeouts, float)
        lo, hi = 0.12, 0.88
        self.p = rng.uniform(lo, hi, (n, 2))
        self.v = np.zeros((n, 2))
        self.g = np.zeros(n)                     # gripper: 0 open, 1 closed
        self.obj = rng.uniform(lo, hi, (n, 2))
        self.place = rng.uniform(lo, hi, (n, 2))
        # keep the three points apart so that every subtask involves real transport
        for _ in range(50):
            bad = (np.linalg.norm(self.obj - self.p, axis=1) < 0.3) | \
                  (np.linalg.norm(self.place - self.obj, axis=1) < 0.3)
            if not bad.any():
                break
            k = bad.sum()
            self.obj[bad] = rng.uniform(lo, hi, (k, 2))
            self.place[bad] = rng.uniform(lo, hi, (k, 2))
        self.holding = np.zeros(n, bool)
        self.phase = np.zeros(n, int)            # 0 pick, 1 place, 2 done
        self.failed = np.zeros(n, bool)
        self.t = 0.0
        self.t_phase_start = np.zeros(n)
        self.sub_time = np.full((n, 2), np.nan)  # completion time of each subtask
        self._armed = np.ones(n, bool)           # gripper re-opened since the last close
        self.missed = np.zeros(n, int)           # closes away from the object

    @property
    def active(self):
        return (self.phase < 2) & ~self.failed

    def target(self):
        return np.where(self.phase[:, None] == 0, self.obj, self.place)

    def obs(self):
        tgt = self.target()
        return np.concatenate([self.p, self.g[:, None], self.holding[:, None].astype(float),
                               self.obj, self.place, tgt - self.p], 1).astype(np.float32)

    def step(self, cmd_p, cmd_g, duration):
        """Hold (cmd_p, cmd_g) for `duration` seconds."""
        w = self.omega
        n_sub = int(round(duration / SUB))
        act = self.active
        for _ in range(n_sub):
            a = w * w * (cmd_p - self.p) - 2.0 * w * self.v
            self.v = np.where(act[:, None], self.v + a * SUB, 0.0)
            self.p = np.where(act[:, None], self.p + self.v * SUB, self.p)
            self.g = np.where(act, self.g + (cmd_g - self.g) * SUB / TAU_G, self.g)
            self.t += SUB
            self._events(act)
            act = self.active

    def _events(self, act):
        # grasp: rising edge of the gripper through 0.7 while picking
        close = act & (self.phase == 0) & self._armed & (self.g > 0.7)
        if close.any():
            ok = close & (np.linalg.norm(self.p - self.obj, axis=1) < R_GRASP)
            self.holding |= ok
            self.missed += (close & ~ok).astype(int)
            self._armed &= ~close
            self._finish(ok, 0)
        self._armed |= self.g < 0.3
        self.obj = np.where(self.holding[:, None], self.p, self.obj)
        # release while placing
        rel = act & (self.phase == 1) & self.holding & (self.g < 0.3)
        if rel.any():
            ok = rel & (np.linalg.norm(self.p - self.place, axis=1) < R_PLACE)
            self.holding &= ~rel
            self.failed |= rel & ~ok          # dropped outside the target
            self._finish(ok, 1)
        # timeouts
        tlim = self.timeouts[np.minimum(self.phase, 1)]
        to = act & (self.phase < 2) & ((self.t - self.t_phase_start) > tlim)
        self.failed |= to

    def _finish(self, mask, k):
        if not mask.any():
            return
        self.sub_time[mask, k] = self.t - self.t_phase_start[mask]
        self.phase = np.where(mask, k + 1, self.phase)
        self.t_phase_start = np.where(mask, self.t, self.t_phase_start)

    def score(self):
        """Green Challenge rule: sum over completed subtasks of 1 - t/T, per episode."""
        s = 1.0 - self.sub_time / self.timeouts[None]
        return np.nan_to_num(s, nan=0.0).sum(1)


# ----------------------------------------------------------------------------------- expert
class Expert:
    """Scripted 'operator' acting in command space at 30 Hz (like a scripted/teleop data
    generator): accelerate toward the current target, cruise at VMAX*tempo, slow down as
    distance/RHO, wait until the hand has settled, then close / open the gripper."""

    def __init__(self, world, tempo):
        self.w = world
        self.tempo = np.broadcast_to(np.asarray(tempo, float), (world.n,)).copy()
        self.c = world.p.copy()
        self.cv = np.zeros_like(world.p)
        self.cg = np.zeros(world.n)
        self.pause = np.zeros(world.n)   # remaining hesitation (teleop style)

    def step(self, rng=None, p_pause=0.0):
        w = self.w
        tgt = w.target()
        d = tgt - self.c
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        vdes = VMAX * self.tempo[:, None] * np.minimum(1.0, dist / RHO) * d / np.maximum(dist, 1e-9)
        hes = self.pause > 0
        vdes[hes] = 0.0
        self.pause = np.maximum(0.0, self.pause - FRAME)
        a = (vdes - self.cv) / FRAME
        an = np.linalg.norm(a, axis=1, keepdims=True)
        amax = AMAX * self.tempo[:, None]
        a = a * np.minimum(1.0, amax / np.maximum(an, 1e-9))
        self.cv = self.cv + a * FRAME
        self.c = self.c + self.cv * FRAME
        near = np.linalg.norm(w.p - tgt, axis=1)
        slow = np.linalg.norm(w.v, axis=1) < SETTLE_V
        pick = (w.phase == 0) & (near < 0.5 * R_GRASP) & slow
        place = (w.phase == 1) & (near < 0.5 * R_PLACE) & slow
        self.cg = np.where(pick, 1.0, self.cg)
        self.cg = np.where(place, 0.0, self.cg)
        # a missed close (hand not at the object): open again and retry
        self.cg = np.where((w.phase == 0) & (self.cg > 0.5) & ~pick & (w.g > 0.9), 0.0, self.cg)
        if rng is not None and p_pause > 0:
            start = (rng.rand(w.n) < p_pause * FRAME) & (self.pause <= 0)
            self.pause = np.where(start, rng.uniform(0.15, 0.5, w.n), self.pause)
        return self.c.copy(), self.cg.copy()


def run_expert(n, seed, tempo=1.0, omega=OMEGA_DEMO, p_pause=0.0, tempo_jitter=0.0, record=True,
               timeouts=(30.0, 30.0)):
    """Roll the expert out in 30 Hz frames. Returns the world and, if record, per-frame arrays
    obs[i], cmd[i] (command issued at frame i), alive[i]."""
    rng = np.random.RandomState(seed)
    w = World(n, rng, omega=omega, timeouts=timeouts)
    tempo = tempo * (1.0 + tempo_jitter * rng.uniform(-1, 1, n))
    ex = Expert(w, tempo)
    obs, cmd, alive = [], [], []
    for _ in range(int(12.0 / FRAME)):
        o = w.obs()
        c, cg = ex.step(rng, p_pause)
        if record:
            obs.append(o)
            cmd.append(np.concatenate([c, cg[:, None]], 1))
            alive.append(w.active.copy())
        w.step(c, cg, FRAME)
        if not w.active.any():
            break
    if not record:
        return w, None
    return w, dict(obs=np.stack(obs, 1), cmd=np.stack(cmd, 1), alive=np.stack(alive, 1), tempo=tempo)


def make_chunks(rec, H=H):
    """(obs at frame i) -> rows j=0..H-1 = command at frame i+1+j, xy as delta to the observed
    hand position (Green: delta actions), gripper absolute. Rows past the end repeat the last."""
    obs, cmd, alive = rec["obs"], rec["cmd"], rec["alive"]
    n, T, _ = obs.shape
    X, Y, EP = [], [], []
    last = alive.sum(1)                     # frames while the episode is running
    for e in range(n):
        L = int(last[e])
        if L < 2:
            continue
        c = cmd[e, :L]
        cpad = np.concatenate([c, np.repeat(c[-1:], H + 1, 0)], 0)
        for i in range(L - 1):
            rows = cpad[i + 1:i + 1 + H].copy()
            rows[:, :2] -= obs[e, i, :2]
            X.append(obs[e, i]); Y.append(rows); EP.append(e)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(EP)


def calibrate_timeouts(seed=123, n=400, q=0.9, margin=1.15):
    """Contest-style T_max: margin x the q-quantile of the expert's own subtask durations
    when executed by the demo robot at tempo 1."""
    w, _ = run_expert(n, seed, record=False)
    d = w.sub_time
    return tuple(float(margin * np.nanquantile(d[:, k], q)) for k in range(2))
