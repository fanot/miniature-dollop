"""Download a few scripted episodes of the three Green Challenge evaluation scenes.

    python greensim/fetch_eval_scenes.py --root DATA --per-scene 6 --out-dir WORK

The scripted part of GreenChallengeData holds digital-twin recordings of exactly the scenes the
leaderboard runs (task strings from green_challenge/kaggle/fetch_data.py): ring, sugar bowl, red soda
can. Writes WORK/episodes_<scene>.txt (comma lists for gch.eval_offline --episodes).
Only data + the three videos of the picked episodes are fetched (< 1 GB for 6 per scene).
"""
import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "SberRoboticsCenter/GreenChallengeData"
SCRIPTED = "scripted/vla"
TASKS = {"ring": "Sber Ring", "sugar": "Raise your right hand, pick up the sugar bowl",
         "soda": "Pick the red soda can up"}
CAMERAS = ("observation.images.cam_head", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
META = ("episodes.jsonl", "info.json", "subtasks.jsonl", "tasks.jsonl", "episodes_stats.jsonl")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--per-scene", type=int, default=6)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name in META:
        try:
            hf_hub_download(REPO, f"{SCRIPTED}/meta/{name}", repo_type="dataset", local_dir=a.root)
        except Exception as e:                      # episodes_stats / subtasks may be absent in some cuts
            print(f"  (meta {name} not downloaded: {type(e).__name__})")
    meta = a.root / SCRIPTED / "meta"
    tasks = {t["task_index"]: t["task"] for t in read_jsonl(meta / "tasks.jsonl")}
    episodes = read_jsonl(meta / "episodes.jsonl")
    info = json.loads((meta / "info.json").read_text())
    chunk = int(info.get("chunks_size", 1000))
    files = []
    for scene, task in TASKS.items():
        eps = [e for e in episodes if task in tasks.get(e.get("task_index"), "") and not e.get("is_problem_episode")]
        random.Random(a.seed).shuffle(eps)
        picked = sorted(e["episode_index"] for e in eps[:a.per_scene])
        if not picked:
            print(f"  {scene}: no episodes found for task {task!r}")
            continue
        (a.out_dir / f"episodes_{scene}.txt").write_text(",".join(map(str, picked)))
        print(f"  {scene}: {len(picked)} episodes {picked} (of {len(eps)})")
        for e in picked:
            files.append(f"{SCRIPTED}/" + info["data_path"].format(episode_chunk=e // chunk, episode_index=e))
            files += [f"{SCRIPTED}/" + info["video_path"].format(episode_chunk=e // chunk, episode_index=e, video_key=c)
                      for c in CAMERAS]
    snapshot_download(REPO, repo_type="dataset", local_dir=a.root, allow_patterns=files, max_workers=16)
    print("FETCH_DONE")


if __name__ == "__main__":
    main()
