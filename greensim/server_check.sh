#!/usr/bin/env bash
# Check the two claims of the README on the real Green Challenge checkpoint, no simulator:
#   A2  does the adaptive-speed event detector (finger travel > 0.01 rad between rows)
#       fire on almost every row of *sampled* plans, i.e. was adaptive15 silently 1x?
#   A   planned tempo: how far the plan travels in 0.5 s relative to the demonstration.
#
#   GC_TOOLKIT=~/green_challenge CKPT=~/ckpt bash greensim/server_check.sh
#
# Optional:
#   CKPT2=~/ckpt_ft1        a second checkpoint for the tempo comparison (A)
#   PYTHON=/path/python     an existing GreenVLA environment (skips the venv install)
#   GREENVLA_ROOT=~/GreenVLA  an existing checkout of greenvla/GreenVLA, branch aij_contest_inference
#   SKIP_ASSETS=1           if inference already ran on this machine (assets are in the HF cache)
#   EPISODES=6              scripted episodes per evaluation scene
#   PRECISION=float16       on T4 / V100 (bf16 is emulated there)
#   ROOT=~/gc_check         working directory
# Needs an NVIDIA GPU with >= 4 GB free. ~15 min setup, then inference: the full eval_offline run took
# ~4 h on Kaggle 2x T4; this one samples only EPISODES per scene, so it scales down with EPISODES.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$HOME/gc_check}
GC_TOOLKIT=${GC_TOOLKIT:?set GC_TOOLKIT to the green_challenge directory}
CKPT=${CKPT:?set CKPT to the checkpoint directory (pretrained_model/ + norm_stats/)}
EPISODES=${EPISODES:-6}
mkdir -p "$ROOT/data" "$ROOT/opt" "$ROOT/work"
# with SKIP_ASSETS the assets are wherever inference already found them, so keep the caller's HF cache
if [ -z "${SKIP_ASSETS:-}" ]; then
  export HF_HUB_CACHE=${HF_HUB_CACHE:-$ROOT/opt/hf-cache} HF_HOME=${HF_HOME:-$ROOT/opt/hf-home}
fi
export TOKENIZERS_PARALLELISM=false

# 1. GreenVLA code and environment (each step skipped when already there)
GREENVLA_ROOT=${GREENVLA_ROOT:-$ROOT/GreenVLA}
[ -d "$GREENVLA_ROOT" ] || git clone -q -b aij_contest_inference https://github.com/greenvla/GreenVLA "$GREENVLA_ROOT"
if [ -n "${PYTHON:-}" ]; then
  PY=$PYTHON
else
  pip install -q uv
  uv python install 3.11
  [ -x "$ROOT/venv/bin/python" ] || uv venv -q --python 3.11 "$ROOT/venv"
  uv pip install -q --python "$ROOT/venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match -r "$GREENVLA_ROOT/pyproject.toml" pyarrow av huggingface_hub
  PY=$ROOT/venv/bin/python
fi
if [ -z "${SKIP_ASSETS:-}" ]; then
  (cd "$GREENVLA_ROOT" && "$PY" tools/repro017799/prepare_assets.py --cache "$ROOT/opt" < /dev/null)
fi

# 2. a patched copy of the green_challenge toolkit (the original stays untouched)
rm -rf "$ROOT/gc" && cp -r "$GC_TOOLKIT" "$ROOT/gc"
(cd "$ROOT/gc" && patch -p1 -s < "$HERE/eval_offline_tempo.patch")
export GREENVLA_ROOT PYTHONPATH="$ROOT/gc:$GREENVLA_ROOT"

# 3. a few scripted episodes of the three evaluation scenes
"$PY" "$HERE/fetch_eval_scenes.py" --root "$ROOT/data" --per-scene "$EPISODES" --out-dir "$ROOT/work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 4. sample plans on demo frames, per scene
CKPTS=(--checkpoint "$CKPT")
[ -n "${CKPT2:-}" ] && CKPTS+=(--checkpoint "$CKPT2")
for scene in ring sugar soda; do
  f="$ROOT/work/episodes_$scene.txt"
  [ -s "$f" ] || continue
  echo "=== $scene ==="
  "$PY" -m gch.eval_offline --data "$ROOT/data/scripted/vla" "${CKPTS[@]}" --episodes "$(cat "$f")" \
      --stride 30 --input-kinematic-space open ${PRECISION:+--precision "$PRECISION"} \
      --out "$ROOT/work/eval_$scene.json" 2>&1 | tee "$ROOT/work/eval_$scene.log" | { grep -E "samples|tempo|finger|torso_arms_neck|Error|error" || true; }
done

# 5. one table
"$PY" - "$ROOT/work" <<'EOF'
import json, sys
from pathlib import Path
w = Path(sys.argv[1])
print("\nscene  checkpoint                      plans kept at 1x  demos kept at 1x  plans flagged  tempo arms (plan/demo)")
for scene in ("ring", "sugar", "soda"):
    p = w / f"eval_{scene}.json"
    if not p.exists():
        continue
    for r in json.loads(p.read_text())["results"]:
        fe, tp = r.get("finger_events", {}), r.get("planned_tempo", {}).get("torso_arms_neck", {})
        print(f"{scene:6s} {Path(r['checkpoint']).name[-30:]:30s}  {fe.get('plans', {}).get('rows_kept_slow', float('nan')):16.2f}"
              f"  {fe.get('demos', {}).get('rows_kept_slow', float('nan')):16.2f}  {fe.get('plans', {}).get('rows_flagged', float('nan')):13.2f}"
              f"  {tp.get('plan_over_demo', float('nan')):.3f}")
print(f"\nfull reports: {w}/eval_*.json")
EOF
