"""Train / dev split of PARTNR's train episodes, kept apart from val (the test set).

- Excluded: every episode in a scene that val also uses, and every episode whose instruction matches a val
  instruction (after stripping whitespace).
- Dev: all remaining episodes of <dev_scenes> held-out train scenes (picked with random.Random(seed) among the scenes
  with at least <min_scene> episodes), and a fixed random sample of <dev_size> of them for evaluation runs.
- Train pool: the remaining episodes of the other scenes.

Entries are [list index, episode_id, scene_id]: planner_demo's +episode_indices takes list indices; its result CSVs
report episode_id (in train they differ).

    python -m muse_habitat.splits [--episodes <dir>] [--out data/splits]
"""
import argparse
import collections
import gzip
import json
import random
from pathlib import Path

EPISODES = Path(__file__).resolve().parents[2] / ".." / "partnr" / "data" / "datasets" / "partnr_episodes" / "v0_0"


def load(path):
    with gzip.open(path) as f:
        return json.load(f)["episodes"]


def make(episodes_dir, dev_scenes=3, dev_size=300, min_scene=1000, seed=0):
    val = load(Path(episodes_dir) / "val.json.gz")
    train = load(Path(episodes_dir) / "train.json.gz")
    val_scenes = {e["scene_id"] for e in val}
    val_instr = {e["instruction"].strip() for e in val}
    rows = [[i, e["episode_id"], e["scene_id"]] for i, e in enumerate(train)]
    shared_scene = [r for r in rows if r[2] in val_scenes]
    shared_instr = [r for r, e in zip(rows, train) if r[2] not in val_scenes and e["instruction"].strip() in val_instr]
    drop = {r[0] for r in shared_scene + shared_instr}
    keep = [r for r in rows if r[0] not in drop]
    sizes = collections.Counter(r[2] for r in keep)
    rng = random.Random(seed)
    dev_set = set(rng.sample(sorted(s for s, n in sizes.items() if n >= min_scene), dev_scenes))
    dev_all = [r for r in keep if r[2] in dev_set]
    dev = sorted(rng.sample(dev_all, dev_size))
    pool = [r for r in keep if r[2] not in dev_set]
    stats = {"train_episodes": len(train), "val_scenes_in_train": sorted(val_scenes & set(r[2] for r in rows)),
             "excluded_shared_scene": len(shared_scene), "excluded_val_instruction": len(shared_instr),
             "dev_scenes": {s: sizes[s] for s in sorted(dev_set)}, "dev_all": len(dev_all), "dev": len(dev),
             "train_pool": len(pool), "train_pool_scenes": len({r[2] for r in pool})}
    return {"stats": stats, "dev": dev, "dev_all": dev_all, "train_pool": pool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(EPISODES))
    ap.add_argument("--out", default="data/splits")
    a = ap.parse_args()
    s = make(a.episodes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in ("dev", "dev_all", "train_pool"):
        with open(out / f"{k}.json", "w") as f:
            json.dump(s[k], f, separators=(",", ":"))
    with open(out / "stats.json", "w") as f:
        json.dump(s["stats"], f, indent=1)
    print(json.dumps(s["stats"], indent=1))


if __name__ == "__main__":
    main()
