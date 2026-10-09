"""Checks that the replica behaves like the organisers' feed and that the task is solvable."""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tempobench import closed_loop as C  # noqa: E402
from tempobench import feed as F  # noqa: E402
from tempobench import policy as P  # noqa: E402
from tempobench import world as W  # noqa: E402


def test_fifteen_live_plans_like_the_organisers_feed():
    # action_feed.py docstring: "with the defaults it settles at **15**" (row_dt 0.025 there)
    for row_dt in (0.025, 1 / 30):
        cfg = F.FeedConfig(speed=W.FRAME / row_dt)
        r = C.rollout(lambda w: P.oracle_plan(w), cfg, n=8, timeouts=(30, 30), max_t=2.5)
        assert r["live_plans_max"] == 15, r["live_plans_max"]


def test_scheduling_plays_rows_on_time():
    n, cfg = 1, F.FeedConfig(blend=False, vlim=1e9, alim=1e9)
    feed = F.Feed(n, cfg)
    feed.reset(np.zeros((1, 2)), np.zeros(1))
    rows = np.zeros((1, W.H, 3))
    rows[0, :, 0] = np.arange(W.H) * 0.01           # row j -> x = 0.01 j
    feed.push(0.0, rows, np.zeros((1, 2)), np.zeros((1, 2)))
    # row j is due at lead + (j+1)*row_dt
    for j in (0, 5, 20):
        t = cfg.lead + (j + 1) * cfg.row_dt
        assert abs(feed.command(t)[0, 0] - 0.01 * j) < 1e-9


def test_old_plans_freeze_past_their_horizon():
    cfg = F.FeedConfig(speed=2.0)                     # horizon 0.03 + 50/60 = 0.863 s < 1.2 s life
    feed = F.Feed(1, cfg)
    feed.reset(np.zeros((1, 2)), np.zeros(1))
    rows = np.zeros((1, W.H, 3))
    rows[0, :, 0] = np.arange(W.H)
    feed.push(0.0, rows, np.zeros((1, 2)), np.zeros((1, 2)))
    val, age, j = feed._sample(1.0)
    assert j[0, 0] == W.H - 1 and val[0, 0, 0] == W.H - 1


def test_mean_plan_age_of_served_feed():
    abar = np.mean([(w * a).sum() / w.sum() for a, _, w in F.mean_plan_age(F.FeedConfig())])
    assert 0.50 < abar < 0.55


def test_oracle_solves_task_at_demo_lag_without_timeouts():
    r = C.rollout(lambda w: P.oracle_plan(w), F.FeedConfig(), n=60, timeouts=(30, 30))
    assert r["subtask_success"] == 1.0


def test_expert_never_misses():
    w, _ = W.run_expert(200, seed=3, record=False)
    assert (~np.isnan(w.sub_time)).all() and w.missed.sum() == 0


def test_formula_tracks_long_cruise():
    """E1 in miniature: steady cruise with the oracle matches the formula within 7 %."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from tempo_experiments import long_world_factory
    orig, init = long_world_factory()
    W.World.__init__ = init
    try:
        for ds, s in [(0.08, 1.0), (0.18, 1.3)]:
            cfg = F.FeedConfig(speed=s, vlim=1e9, alim=1e9)
            r = C.rollout(lambda w: P.oracle_plan(w), cfg, n=20, omega=W.omega_for_lag(ds),
                          timeouts=(30, 30), max_t=4.0)
            pred = F.predicted_tempo_ratio(cfg, ds)
            assert abs(r["tempo_ratio_abs"] - pred) / pred < 0.07, (ds, s, r["tempo_ratio_abs"], pred)
    finally:
        W.World.__init__ = orig
