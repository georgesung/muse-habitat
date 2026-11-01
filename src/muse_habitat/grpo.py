"""GRPO on PARTNR trajectories: groups, advantages, packed sequences and the clipped policy-gradient loss.

A group is <samples> rollouts of one episode (scripts/partnr_groups.sh: runs/<name>/traj_k<k>/ and the result CSVs in
runs/<name>/k<k>_s<j>/results/). Each rollout's reward is shared by its two agents; the advantage of a rollout is its
reward minus the group mean, divided by the group std (groups with equal rewards carry no signal and are dropped).

Training data are the agents' own generated tokens only. One agent's turns pack into sequences (muse_habitat.
turn_packing): the last turn's context once, then each turn's tokens at the positions they were sampled at; branches
longer than a token budget are cut into consecutive chunks of turns, each packed against its own last context.

Loss per token (standard clipped objective, importance ratio against the sampling policy's logged log-probs):
    -min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A),   r = exp(logp - logp_sampled)
summed over the batch's tokens and divided by the batch's token count.
"""
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Turn:
    prompt: list  # prompt token ids (with BOS), as vLLM saw them
    tokens: list  # generated token ids (thinking + call + stop string)
    logprobs: list  # sampling policy's log-prob of each generated token


@dataclass
class Rollout:
    sample: int
    episode_id: str
    reward: float = math.nan
    success: float = math.nan
    agents: dict = field(default_factory=dict)  # agent uid -> [Turn] in turn order
    advantage: float = 0.0
    models: set = field(default_factory=set)  # served model names (policy versions) that sampled its turns


def reward_of(row, kind):
    if kind == "success":
        return float(row["task_state_success"])
    if kind == "percent":
        return float(row["task_percent_complete"])
    if kind == "success+percent":  # success, plus partial credit for the propositions done
        return float(row["task_state_success"]) + 0.5 * float(row["task_percent_complete"])
    raise ValueError(kind)


def _add(out, k, r):
    if r.get("token_ids") is None or r.get("grammar"):  # constrained fallback turns: context only
        return
    ro = out.setdefault((k, r["episode_id"]), Rollout(k, r["episode_id"]))
    ro.agents.setdefault(r["agent"], []).append((r["turn"], Turn(r["prompt_token_ids"], r["token_ids"], r["logprobs"])))
    ro.models.add(r.get("model"))


def _finish(out, run_dir, reward, keys=None):
    """Rewards from the result CSVs, turns in order; drops rollouts without a result row (PARTNR crashed)."""
    for k in sorted({key[0] for key in out}):
        for f in sorted(Path(run_dir).glob(f"k{k}_s*/results/episode_result_log.csv")):
            for row in csv.DictReader(open(f)):
                ro = out.get((k, row["episode_id"]))
                if ro is not None:
                    ro.reward, ro.success = reward_of(row, reward), float(row["task_state_success"])
    done = {}
    for key, ro in out.items():
        if keys is not None and key not in keys:
            continue
        if isinstance(next(iter(ro.agents.values()), [None])[0], tuple):
            ro.agents = {a: [t for _, t in sorted(ts, key=lambda x: x[0])] for a, ts in ro.agents.items()}
        if not math.isnan(ro.reward):
            done[key] = ro
    return done


def load_rollouts(run_dir, reward="success"):
    """{(sample, episode_id): Rollout} with rewards; rollouts without a result row (PARTNR crashed) are dropped."""
    run_dir = Path(run_dir)
    out = {}
    for traj in sorted(run_dir.glob("traj_k*")):
        k = int(traj.name[len("traj_k"):])
        for f in sorted(traj.glob("*.jsonl")):
            for line in open(f):
                _add(out, k, json.loads(line))
    return _finish(out, run_dir, reward)


class TrajReader:
    """Incremental reader of a running groups run: parses only lines appended since the last poll, holds the turns of
    episodes not taken yet."""

    def __init__(self, run_dir, reward="success"):
        self.run_dir, self.reward = Path(run_dir), reward
        self.offsets, self.out = {}, {}

    def poll(self):
        for traj in sorted(self.run_dir.glob("traj_k*")):
            k = int(traj.name[len("traj_k"):])
            for f in sorted(traj.glob("*.jsonl")):
                with open(f) as fh:
                    fh.seek(self.offsets.get(f, 0))
                    while True:
                        line = fh.readline()
                        if not line.endswith("\n"):  # partial last line: read it next time
                            break
                        self.offsets[f] = fh.tell()
                        _add(self.out, k, json.loads(line))

    def take(self, episode_ids):
        """{(sample, episode_id): Rollout} of these episodes (with rewards), removed from memory."""
        self.poll()
        keys = {key for key in self.out if key[1] in set(episode_ids)}
        got = _finish({key: self.out[key] for key in keys}, self.run_dir, self.reward)
        for key in keys:
            del self.out[key]
        return got


def groups(rollouts, min_size=2, eps=1e-6):
    """Group rollouts by episode, set advantages; returns the groups with any reward spread."""
    by = defaultdict(list)
    for ro in rollouts.values():
        by[ro.episode_id].append(ro)
    out = []
    for eid, g in sorted(by.items()):
        if len(g) < min_size:
            continue
        r = torch.tensor([x.reward for x in g])
        if r.std() < eps:
            continue
        for x, a in zip(g, ((r - r.mean()) / (r.std() + eps)).tolist()):
            x.advantage = a
        out.append(g)
    return out


def chunks(turns, budget):
    """Consecutive runs of turns whose packed length (last context + generated tokens) stays within budget."""
    out, cur = [], []
    for t in turns:
        if cur and len(t.prompt) + sum(len(x.tokens) for x in cur) + len(t.tokens) > budget:
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


@dataclass
class Pack:
    chain: list
    turns: list
    advantage: float

    @property
    def n_tokens(self):
        return len(self.chain) + sum(len(t.tokens) for t in self.turns)

    @property
    def n_gen(self):
        return sum(len(t.tokens) for t in self.turns)


def packs(group_list, budget=12288):
    """All packs of the given groups: one per agent per rollout, more for branches over the budget."""
    out = []
    for g in group_list:
        for ro in g:
            for turns in ro.agents.values():
                for c in chunks(turns, budget):
                    for a, b in zip(c, c[1:]):
                        assert b.prompt[: len(a.prompt)] == a.prompt, "history is not append-only"
                    out.append(Pack(c[-1].prompt, c, ro.advantage))
    return out


def pack_objective(lp, ent, pack, eps_low=0.2, eps_high=0.28):
    """Sum of the clipped objective over a pack's generated tokens (to minimize), and stats."""
    old = torch.tensor([x for t in pack.turns for x in t.logprobs], device=lp.device, dtype=torch.float32)
    ratio = torch.exp(lp - old)
    a = pack.advantage
    obj = torch.minimum(ratio * a, ratio.clamp(1 - eps_low, 1 + eps_high) * a)
    r = ratio.detach()
    stats = {"tokens": lp.numel(), "logp": lp.detach().sum().item(), "entropy": ent.sum().item(),
             "ratio_dev": (r - 1).abs().sum().item(), "clipped": ((r < 1 - eps_low) | (r > 1 + eps_high)).sum().item()}
    return -obj.sum(), stats


def train_step(model, opt, pack_list, tc, max_grad_norm=1.0, device="cuda", **obj_kw):
    """One optimizer step over the packs (one forward and backward each, gradients accumulated); the loss is
    normalized by the step's generated-token count. The pack stays active through the backward, where gradient
    checkpointing recomputes attention."""
    from muse_habitat.turn_packing import active_pack, build_pack, packed_logps
    n_tok = sum(p.n_gen for p in pack_list)
    agg = defaultdict(float)
    for p in pack_list:
        pk = build_pack(torch.tensor(p.chain), [(len(t.prompt), torch.tensor(t.tokens)) for t in p.turns],
                        tc.sliding_window, device)
        with active_pack(pk):
            lp, ent = packed_logps(model, pk, tc.final_logit_softcapping, 1.0, tc.output_multiplier)
            loss, st = pack_objective(lp, ent, p, **obj_kw)
            (loss / n_tok).backward()
        for k, v in st.items():
            agg[k] += v
        agg["loss"] += loss.item() / n_tok
    params = [q for q in model.parameters() if q.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    t = agg["tokens"]
    return {"tokens": int(t), "packs": len(pack_list), "loss": agg["loss"], "grad_norm": grad_norm,
            "mean_logp": agg["logp"] / t, "entropy": agg["entropy"] / t, "ratio_dev": agg["ratio_dev"] / t,
            "clip_frac": agg["clipped"] / t}
