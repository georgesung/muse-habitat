"""Streaming GRPO on PARTNR: rollouts never stop, the trainer steps whenever a group with a reward spread completes.

- Rollouts: <samples> x <shards> long-lived planner_demo processes (C1 low, temperature 1, trajectory logging). Shard
  j's episodes (a seeded shuffle of the train pool) go to all <samples> processes of that shard in the same order
  (Habitat's episode shuffle and scene grouping are turned off), so a group (all samples of one episode) completes
  once every process of its shard has passed it. Each LLM call reads the policy (LoRA adapter name) from
  <run>/policy, so processes follow updates from their next turn on; the logged log-probs are the sampling policy's.
- Trainer (this process, one GPU): bf16 Glimmer + LoRA (muse_habitat.grpo: packed sequences, clipped objective with the
  importance ratio against the logged log-probs). After each step: save <run>/adapters/p<n>, load it into the vLLM
  server, point <run>/policy at it, unload adapters older than the last <keep>.
- Restart: --init <adapter dir> starts from that adapter (served as p<start-step> from the start), --init-opt <file>
  from that optimizer state (<run>/optimizer.pt holds the latest one); --start-step <n> continues the step count and
  adapter names (p<n+1>, ...) of the run the adapter came from.
- Logs: <run>/train_log.jsonl (one line per step), <run>/groups.jsonl (every completed group, trained or not).

The vLLM servers (scripts/glimmer_serve_lora.sh, with --max-lora-rank >= rank) must be up on <ports> (comma-separated;
rollout process n uses port n mod len, adapters are loaded into all of them). Servers on another machine (ports
forwarded here, scripts/tunnel.sh): --remote <host>:<dir> rsyncs each adapter to <dir>/<name> there before loading it.

    CUDA_VISIBLE_DEVICES=<trainer gpu> envs/trl/bin/python -I scripts/grpo_run.py <name> --port 8021[,8022,...] \
        --rollout-gpus "0 1 1" [--samples 8 --shards 3 --episodes 300 --steps 20]
"""
import argparse
import csv
import json
import os
import random
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import torch

from muse_habitat.grpo import groups as make_groups
from muse_habitat.grpo import TrajReader, packs, train_step

ROOT = Path(__file__).resolve().parents[1]
from muse_habitat.models import GLIMMER as MODEL
TARGETS = r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def post(port, path, body, tries=10):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for i in range(tries):  # survives a server or tunnel hiccup
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read().decode()
        except OSError as e:
            if i == tries - 1:
                raise
            print(f"POST {port}{path} failed ({e}), retrying", flush=True)
            time.sleep(30)


def write_atomic(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def start_rollouts(run, a, shards):
    """One planner_demo per (sample, shard); returns {(k, j): Popen} (each in its own process group)."""
    gpus = a.rollout_gpus.split()
    procs, n = {}, 0
    for k in range(a.samples):
        (run / f"traj_k{k}").mkdir(parents=True, exist_ok=True)
        for j, eps in enumerate(shards):
            over = [f"+episode_indices=[{','.join(str(e[0]) for e in eps)}]",
                    "habitat.environment.iterator_options.shuffle=False",
                    "habitat.environment.iterator_options.group_by_scene=False",
                    "habitat.environment.iterator_options.max_scene_repeat_steps=-1"]
            for ag in (0, 1):
                p = f"evaluation.agents.agent_{ag}.planner"
                over += [f"llm@{p}.plan_config.llm=glimmer_vllm_tools1",
                         f"instruct@{p}.plan_config.instruct=zero_shot_native_tools",
                         f"{p}._target_=muse_habitat.native_tools.NativeToolsPlanner",
                         f"{p}.plan_config.llm.port={a.ports[n % len(a.ports)]}",
                         f"{p}.plan_config.llm.generation_params.temperature={a.temperature}"]
            env = {**os.environ, "PARTNR_SPLIT": "train", "MUSE_TRAJ_DIR": str(run / f"traj_k{k}"),
                   "MUSE_POLICY_FILE": str(run / "policy")}
            env.pop("CUDA_VISIBLE_DEVICES", None)
            cmd = [str(ROOT / "scripts" / "partnr_run.sh"), f"{a.name}/k{k}_s{j}", gpus[n % len(gpus)],
                   "baselines/decentralized_zero_shot_react_summary.yaml", *over]
            procs[(k, j)] = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
            n += 1
    return procs


def done_ids(run, k, j):
    """Episode ids with a result row in process (k, j)."""
    f = run / f"k{k}_s{j}" / "results" / "episode_result_log.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return [r["episode_id"] for r in csv.DictReader(fh)]


def completed_groups(run, a, shards, consumed):
    """Episode ids whose group is complete: every process of the shard has a row for it or has moved past it (PARTNR
    skips episodes that crash). In shard order; consumed ones left out."""
    out = []
    for j, eps in enumerate(shards):
        order = [e[1] for e in eps]
        pos = {eid: i for i, eid in enumerate(order)}
        progress = []
        for k in range(a.samples):
            ids = done_ids(run, k, j)
            progress.append((set(ids), max((pos[i] for i in ids if i in pos), default=-1)))
        for eid in order:
            if eid in consumed:
                continue
            if all(eid in ids or last > pos[eid] for ids, last in progress):
                out.append(eid)
            else:
                break  # later episodes of this shard cannot be complete before this one in most processes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--port", default="8021", help="vLLM server port(s), comma-separated")
    ap.add_argument("--rollout-gpus", default="0")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300, help="episodes drawn from the train pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--groups-per-step", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reward", default="success")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--budget", type=int, default=12288)
    ap.add_argument("--keep", type=int, default=3, help="adapters kept loaded in vLLM")
    ap.add_argument("--remote", help="<host>:<dir> where the servers read adapters (rsync), if not this machine")
    ap.add_argument("--rsh", default="ssh", help="rsync's remote shell for --remote")
    ap.add_argument("--init", help="LoRA adapter dir to start from")
    ap.add_argument("--init-opt", help="optimizer state to start from (a run's optimizer.pt)")
    ap.add_argument("--start-step", type=int, default=0, help="steps already taken (the --init adapter's number)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="with --remote: delete local adapters once unloaded, except every <n>th (0: keep all)")
    a = ap.parse_args()
    a.ports = [int(x) for x in a.port.split(",")]
    run = ROOT / "runs" / a.name
    (run / "adapters").mkdir(parents=True, exist_ok=True)
    pool = json.load(open(ROOT / "data" / "splits" / "train_pool.json"))
    eps = random.Random(a.seed).sample(pool, a.episodes)
    shards = [eps[j::a.shards] for j in range(a.shards)]
    json.dump({"args": vars(a), "shards": shards}, open(run / "config.json", "w"))

    def publish(name, path):
        """Load adapter <path> into every server as <name> (copied there first with --remote)."""
        if a.remote:
            subprocess.run(["rsync", "-a", "--mkpath", "-e", a.rsh, f"{path}/", f"{a.remote}/{name}/"], check=True)
            path = f"{a.remote.split(':', 1)[1]}/{name}"
        for port in a.ports:
            post(port, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})

    loaded = []
    if a.init:
        publish(f"p{a.start_step}", Path(a.init).resolve())
        loaded.append(f"p{a.start_step}")
    write_atomic(run / "policy", (f"p{a.start_step}" if a.init else "glimmer") + "\n")
    t_start = time.time()
    log = open(run / "train_log.jsonl", "a")
    glog = open(run / "groups.jsonl", "a")

    procs = start_rollouts(run, a, shards)
    print(f"{len(procs)} rollout processes started", flush=True)

    def stop(*_):
        for p in procs.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if a.init:
        model = PeftModel.from_pretrained(base, a.init, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=TARGETS))
    model.train()
    tc = model.config.get_text_config()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    if a.init_opt:
        opt.load_state_dict(torch.load(a.init_opt, map_location="cuda:0"))
    print(f"trainer ready {time.time() - t_start:.0f} s", flush=True)

    consumed, pending, step = set(), [], a.start_step
    reader = TrajReader(run, a.reward)
    try:
        while step < a.steps and time.time() - t_start < a.hours * 3600:
            reader.poll()
            ready = completed_groups(run, a, shards, consumed)
            if ready:
                ros = reader.take(ready)
                for eid in ready:
                    consumed.add(eid)
                    g = [ro for (k, e), ro in ros.items() if e == eid]
                    rewards = [ro.reward for ro in g]
                    usable = make_groups({(ro.sample, eid): ro for ro in g})
                    rec = {"time": time.time() - t_start, "episode_id": eid, "rollouts": len(g), "rewards": rewards,
                           "policies": sorted({t for ro in g for t in ro.models}), "trained": bool(usable)}
                    glog.write(json.dumps(rec) + "\n")
                    glog.flush()
                    pending += usable
            if len(pending) < a.groups_per_step:
                if all(p.poll() is not None for p in procs.values()):
                    print("all rollouts finished", flush=True)
                    break
                time.sleep(20)
                continue
            batch, pending = pending[: a.groups_per_step], pending[a.groups_per_step:]
            pl = packs(batch, a.budget)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            st = train_step(model, opt, pl, tc)
            st.update(step=step, time=time.time() - t_start, seconds=time.time() - t0,
                      peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                      episodes=[g[0].episode_id for g in batch], rewards=[[r.reward for r in g] for g in batch],
                      policies=sorted({m for g in batch for ro in g for m in ro.models}))
            name = f"p{step + 1}"
            model.save_pretrained(run / "adapters" / name)
            torch.save(opt.state_dict(), run / "optimizer.pt.tmp")
            os.replace(run / "optimizer.pt.tmp", run / "optimizer.pt")
            publish(name, run / "adapters" / name)
            write_atomic(run / "policy", name + "\n")
            loaded.append(name)
            while len(loaded) > a.keep:  # in-flight turns of older policies finish within the keep window
                old = loaded.pop(0)
                for port in a.ports:
                    post(port, "/v1/unload_lora_adapter", {"lora_name": old})
                if a.remote and a.save_every and int(old[1:]) > a.start_step and int(old[1:]) % a.save_every:
                    shutil.rmtree(run / "adapters" / old)
            st["policy"] = name
            print(json.dumps(st), flush=True)
            log.write(json.dumps(st) + "\n")
            log.flush()
            step += 1
    finally:
        for p in procs.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        print(f"stopped after {step} steps, {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
