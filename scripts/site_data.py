"""Data for the project page (docs/): per val episode, each model's outcome and turn-by-turn transcript.

Reads the val runs' shard logs (run.log: the planner's thinking and tool call per turn, the executed action's agent and
its result) and result CSVs, the recheck runs, and the val dataset (instruction, house). Writes docs/data/episodes.json
(one row per episode) and docs/data/ep/<id>.json (transcripts).

    python3 scripts/site_data.py <original model val run dirs glob>
"""
import argparse
import csv
import glob
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# rl: the released adapter (run dirs say p173); rl_longer: 100 more updates (p273). README "Details" explains the names.
RUNS = {"rl": "runs/val_r3p173_h*_s*", "rl_longer": "runs/val_r5p273_h*_s*"}
RECHECK = {"original": "runs/recheck_glimmer_h*_s*", "rl": "runs/recheck_p173_h*_s*",
           "rl_longer": "runs/recheck_p273_h*_s*"}
MAX_RESULT = 400


def results(dirs):
    out = {}
    for d in dirs:
        f = Path(d) / "results/episode_result_log.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                out[int(r["episode_id"])] = {
                    "success": float(r["task_state_success"]) == 1, "percent": round(float(r["task_percent_complete"]), 3),
                    "steps": int(float(r["sim_step_count"])), "replans": [int(float(r["replanning_count_0"])),
                                                                          int(float(r["replanning_count_1"]))]}
    return out


def transcripts(dirs):
    """{episode: [turn]}; turn = {agent, thinking, call, result, fallback}. The agent comes from the executed action;
    turns without one (Done) go to the agent that is free (its last action has returned). Results: the world graph's
    log of a finished action, else (immediate failures) the result echoed into the next prompt."""
    out = {}
    for d in dirs:
        f = Path(d) / "run.log"
        if not f.exists():
            continue
        ep = turns = None
        think, cur, fallback = None, None, False
        busy = {}  # agent -> pending turn awaiting its result
        last_free = [0, 1]
        last_exec, echoed = None, False  # echoed: the next prompt echo repeats a logged result

        def close():
            nonlocal cur
            if cur is not None and cur["agent"] is None:
                free = [a for a in last_free if a not in busy]
                cur["agent"] = free[-1] if free else None
            cur = None

        for raw in open(f, errors="replace"):
            line = ANSI.sub("", raw.rstrip("\n"))
            m = re.match(r"Episode (\d+)$", line)
            if m:
                close()
                ep = int(m.group(1))
                turns = out[ep] = []
                think, cur, fallback, busy, last_free, last_exec, echoed = None, None, False, {}, [0, 1], None, False
                continue
            if turns is None:
                continue
            if line.startswith("[NativeToolsPlanner] "):
                body = line[len("[NativeToolsPlanner] "):]
                m = re.match(r"thinking(?: \(\d+ tokens\))?: ?(.*)$", body)
                if m:
                    close()
                    think = [m.group(1)]
                    continue
                if body.startswith("single call: no tool call"):
                    close()
                    fallback, think = True, None
                    continue
                m = re.match(r"(?:single call|call): (.*) \(\d+ tokens\)", body)
                if m:
                    close()
                    cur = {"agent": None, "thinking": "\n".join(think or []).strip(), "call": m.group(1),
                           "result": None}
                    if fallback:
                        cur["fallback"] = True
                    turns.append(cur)
                    think, fallback = None, False
                    continue
            if think is not None:
                think.append(line)
                continue
            m = re.match(r"\[DEBUG\] Now Executing: \{(\d+): \('([^']*)'", line)
            if m and cur is not None:
                cur["agent"] = int(m.group(1))
                busy[cur["agent"]] = cur
                last_exec, cur = cur, None
                continue
            if raw.startswith("<|start|>user<|message|>Result: "):
                if echoed:
                    echoed = False
                elif last_exec is not None and last_exec["result"] is None:
                    r = line[len("<|start|>user<|message|>Result: "):].split("<|eot|>")[0]
                    last_exec["result"] = r if len(r) <= MAX_RESULT else r[:MAX_RESULT] + " ..."
                    busy.pop(last_exec["agent"], None)
                continue
            m = re.search(r"agent_uid=(\d+): high_level_action=\('([^']*)'.*action_response=(['\"])(.*)\3$", line)
            if m and not m.group(2).startswith("AtomicSkills"):
                a = int(m.group(1))
                t = busy.pop(a, None)
                if t is not None and t["result"] is None:
                    r = m.group(4).replace("\\n", "\n")
                    t["result"] = r if len(r) <= MAX_RESULT else r[:MAX_RESULT] + " ..."
                    echoed = True
                if a in last_free:
                    last_free.remove(a)
                last_free.append(a)
        close()
    for ts in out.values():
        for t in ts:
            if t["result"] is None:
                del t["result"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", help="glob of the original model's val shard dirs")
    ap.add_argument("--dataset", default=str(ROOT.parent / "partnr/data/datasets/partnr_episodes/v0_0/val.json.gz"))
    ap.add_argument("--out", default=str(ROOT / "docs/data"))
    a = ap.parse_args()
    eps = json.load(gzip.open(a.dataset))["episodes"]
    houses = {s: i + 1 for i, s in enumerate(sorted({e["scene_id"] for e in eps}))}
    dirs = {"original": sorted(glob.glob(a.original)), **{m: sorted(glob.glob(str(ROOT / g))) for m, g in RUNS.items()}}
    dirs = {m: [d for d in ds if Path(d).is_dir()] for m, ds in dirs.items()}
    res = {m: results(ds) for m, ds in dirs.items()}
    tr = {m: transcripts(ds) for m, ds in dirs.items()}
    rdirs = {m: [d for d in sorted(glob.glob(str(ROOT / g))) if Path(d).is_dir()] for m, g in RECHECK.items()}
    rres = {m: results(ds) for m, ds in rdirs.items()}
    rtr = {m: transcripts(ds) for m, ds in rdirs.items()}
    rechecked = set(json.load(open(ROOT / "runs/val_recheck_episodes.json"))["disagree"]) | \
        set(json.load(open(ROOT / "runs/val_recheck_episodes.json"))["agree_sample"])
    out = Path(a.out)
    (out / "ep").mkdir(parents=True, exist_ok=True)
    rows, gaps = [], []
    for i, e in enumerate(eps):
        s = lambda m, R=res: (R[m].get(i) or {}).get("success", False)  # crash = fail
        row = {"id": i, "instruction": e["instruction"].strip(), "house": houses[e["scene_id"]],
               "scene": e["scene_id"]}
        for m in ("original", "rl", "rl_longer"):
            row[m] = res[m].get(i) or {"success": None, "crashed": True}
        if i in rechecked:
            row["recheck"] = {m: (rres[m].get(i) or {"success": None, "crashed": True}) for m in RECHECK}
            o2 = [s("original"), s("original", rres)]
            p2 = [s("rl"), s("rl", rres)]
            if o2 == [False, False] and p2 == [True, True]:
                cat = "rl_win"
            elif o2 == [True, True] and p2 == [False, False]:
                cat = "rl_loss"
            elif all(o2 + p2) and s("rl_longer") and s("rl_longer", rres):
                cat = "all_success"
            elif not any(o2 + p2 + [s("rl_longer"), s("rl_longer", rres)]):
                cat = "all_fail"
            else:
                cat = "noisy"
        else:
            v = [s("original"), s("rl"), s("rl_longer")]
            cat = "all_success" if all(v) else "all_fail" if not any(v) else "disagree"
        row["category"] = cat
        rows.append(row)
        ep = {"id": i, "instruction": row["instruction"], "runs": {}}
        for m in ("original", "rl", "rl_longer"):
            if i in tr[m]:
                ep["runs"][m] = tr[m][i]
            else:
                gaps.append((m, i))
        if i in rechecked:
            for m in RECHECK:
                if i in rtr[m]:
                    ep["runs"][m + " (re-run)"] = rtr[m][i]
        json.dump(ep, open(out / "ep" / f"{i}.json", "w"), separators=(",", ":"))
    json.dump({"houses": len(houses), "episodes": rows}, open(out / "episodes.json", "w"), separators=(",", ":"))
    cats = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("categories", cats)
    print("success", {m: sum(1 for r in rows if r[m].get("success")) for m in ("original", "rl", "rl_longer")})
    print("transcript gaps", len(gaps), gaps[:20])


if __name__ == "__main__":
    main()
