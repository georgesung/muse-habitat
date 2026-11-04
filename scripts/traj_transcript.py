"""Readable transcript of one PARTNR episode from its trajectory logs (MUSE_TRAJ_DIR, native tools planner): per turn,
the agent (robot / human), seconds since the first turn, the model's thinking, its tool call and the tool's result
(taken from that agent's next prompt). Markdown on stdout.

    python3 scripts/traj_transcript.py <traj dir or jsonl> [--episode <id>]
"""
import argparse
import json
import re
from pathlib import Path

AGENTS = {0: "Robot", 1: "Human"}


def turn_parts(text):
    """(thinking, tool name, {param: value}) of one generated turn."""
    m = re.search(r"to=self<\|message\|>(.*?)<\|eom\|>", text, re.S)
    thinking = m.group(1).strip() if m else ""
    m = re.search(r'<atem:invoke name="([^"]+)">(.*?)(?:</atem:invoke>|$)', text, re.S)
    if not m:
        return thinking, None, {}
    params = dict(re.findall(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', m.group(2), re.S))
    return thinking, m.group(1), params


def last_tool_output(prompt):
    outs = re.findall(r"<tool_output[^>]*>\n?(.*?)</tool_output>", prompt, re.S)
    return outs[-1].strip() if outs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--episode")
    a = ap.parse_args()
    p = Path(a.traj)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    recs = [json.loads(l) for f in files for l in open(f)]
    if a.episode is not None:
        recs = [r for r in recs if str(r.get("episode_id")) == a.episode]
    recs.sort(key=lambda r: r["time"])
    if not recs:
        raise SystemExit("no turns")
    t0 = recs[0]["time"]
    # the result of an agent's call is in that agent's next prompt
    result = {}
    last = {}
    for i, r in enumerate(recs):
        if r["agent"] in last:
            result[last[r["agent"]]] = last_tool_output(r["prompt"])
        last[r["agent"]] = i
    print(f"# Episode {recs[0].get('episode_id')} ({recs[0]['model']})\n")
    for i, r in enumerate(recs):
        thinking, tool, params = turn_parts(r["text"])
        call = f"{tool}({', '.join(params.values())})" if tool else "(no tool call)"
        print(f"**{r['time'] - t0:7.1f} s  {AGENTS.get(r['agent'], r['agent'])}, turn {r['turn']}: `{call}`**\n")
        if thinking:
            print("> " + thinking.replace("\n", "\n> ") + "\n")
        if result.get(i):
            print(result[i].replace("\n", "  \n") + "\n")


if __name__ == "__main__":
    main()
