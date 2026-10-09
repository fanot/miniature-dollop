"""Generate a randomised trial sheet for the real-robot grounding-bypass protocol.

Example (positions in cm in the table frame used for the demos):
    python real_robot/make_trial_sheet.py \
        --objects tape pliers screwdriver --train_positions my_demo_positions.csv \
        --table 60 40 --grid 4 3 --reps 2 --out trials.csv

The sheet is filled in by the operator during the session (columns outcome / grabbed /
final_x_cm / final_y_cm / notes) and then analysed with analyze_trials.py.
See docs/real_robot_protocol.md for what each probe tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import random


def load_usual_places(path):
    """CSV with columns object,x_cm,y_cm (one row per demonstration). Returns mean place per object."""
    acc = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            acc.setdefault(row["object"], []).append((float(row["x_cm"]), float(row["y_cm"])))
    return {k: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", required=True)
    ap.add_argument("--train_positions", required=True)
    ap.add_argument("--table", type=float, nargs=2, default=[60, 40], help="usable table area W H in cm")
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--lambdas", type=float, nargs="*", default=[0.0, 0.5, 1.0],
                    help="guidance scales for the counterfactual-anchor probe (Green-VLA JPM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="trials.csv")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    usual = load_usual_places(a.train_positions)
    W, H = a.table
    gx, gy = a.grid
    cells = [((i + 0.5) * W / gx, (j + 0.5) * H / gy) for i in range(gx) for j in range(gy)]
    rows = []

    def add(probe, target, tx, ty, distractors, extra=None):
        rows.append({"probe": probe, "instruction": f"pick the {target}", "target": target,
                     "target_x_cm": round(tx, 1), "target_y_cm": round(ty, 1),
                     "distractors": json.dumps(distractors), **(extra or {})})

    for obj in a.objects:
        others = [o for o in a.objects if o != obj]
        # P1 position grid: the target alone at every cell (others at their usual places)
        for (x, y) in cells:
            for _ in range(a.reps):
                add("position_grid", obj, x, y, {o: usual.get(o) for o in others})
        # P2 swap: target and one distractor exchange their usual places
        for o in others:
            for _ in range(a.reps):
                add("swap", obj, *usual[o], {o: usual[obj], **{q: usual.get(q) for q in others if q != o}})
        # P3 empty: target removed, everything else at the usual places
        for _ in range(a.reps):
            add("empty", obj, *usual[obj], {o: usual.get(o) for o in others}, {"notes": "TARGET REMOVED"})
        # P5 counterfactual anchor: JPM target point is overridden to a distractor
        for lam in a.lambdas:
            o = rnd.choice(others)
            for _ in range(a.reps):
                add("anchor_cf", obj, *usual[obj], {q: usual.get(q) for q in others},
                    {"anchor_override": o, "guidance_lambda": lam})
    rnd.shuffle(rows)
    cols = ["trial_id", "probe", "instruction", "target", "target_x_cm", "target_y_cm", "distractors",
            "anchor_override", "guidance_lambda", "outcome", "grabbed", "final_x_cm", "final_y_cm", "notes"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"trial_id": i, **{c: r.get(c, "") for c in cols if c != "trial_id"}})
    print(f"wrote {len(rows)} trials to {a.out}")
    print("outcome vocabulary: success | wrong_object | miss | collision | timeout | air_at_usual_place")


if __name__ == "__main__":
    main()
