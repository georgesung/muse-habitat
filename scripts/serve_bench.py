"""Rollout-like decode throughput of a Glimmer server: real PARTNR turn prompts from trajectory logs, temperature 1,
the planner's stop string, <conc> requests kept in flight for <seconds>. Reports generated tokens/s and per-turn
latency.

    python scripts/serve_bench.py <traj dir or jsonl> --port 8021 --model glimmer --conc 8,16,24 [--seconds 120]
"""
import argparse
import json
import random
import threading
import time
import urllib.request
from pathlib import Path


def complete(port, model, prompt, seed):
    body = {"model": model, "prompt": prompt, "max_tokens": 700, "temperature": 1.0, "seed": seed,
            "stop": ["</atem:function_calls>"], "skip_special_tokens": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="glimmer")
    ap.add_argument("--conc", default="8,16,24")
    ap.add_argument("--seconds", type=float, default=120)
    a = ap.parse_args()
    p = Path(a.traj)
    files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    prompts = [json.loads(l)["prompt_token_ids"] for f in files for l in open(f)]
    random.Random(0).shuffle(prompts)
    print(f"{len(prompts)} prompts, median {sorted(map(len, prompts))[len(prompts) // 2]} tokens", flush=True)
    for conc in map(int, a.conc.split(",")):
        stop_at = time.time() + a.seconds
        lock, stats, it = threading.Lock(), [], iter(range(10**9))

        def worker():
            while time.time() < stop_at:
                with lock:
                    i = next(it)
                t0 = time.time()
                n = complete(a.port, a.model, prompts[i % len(prompts)], i)
                with lock:
                    stats.append((t0, time.time(), n))
        t_start = time.time()
        ts = [threading.Thread(target=worker) for _ in range(conc)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.time() - t_start
        toks = sum(n for _, _, n in stats)
        lat = sorted(e - s for s, e, _ in stats)
        print(json.dumps({"model": a.model, "conc": conc, "turns": len(stats), "tok_per_s": toks / wall,
                          "turns_per_min": 60 * len(stats) / wall, "mean_tokens": toks / len(stats),
                          "latency_median_s": lat[len(lat) // 2], "latency_p90_s": lat[int(0.9 * (len(lat) - 1))]}),
              flush=True)


if __name__ == "__main__":
    main()
