"""Prefix-shared packing (muse_habitat.turn_packing) vs per-turn samples on Glimmer 30B, on one logged branch
(episode, agent) of real PARTNR turns: log-probs of the generated tokens, LoRA gradients of a token-weighted loss,
time and peak memory of each way.

    envs/trl/bin/python -I scripts/pack_check.py <traj jsonl> <episode_id> <agent> [--max-turns N]
"""
import argparse
import json
import time

import torch

from muse_habitat.turn_packing import active_pack, build_pack, packed_logps
from muse_habitat.models import GLIMMER as MODEL
TARGETS = r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("episode")
    ap.add_argument("agent", type=int)
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--weights", choices=["ones", "random"], default="ones",
                    help="per-token loss weights: ones (one advantage per branch, as GRPO) or random signs")
    a = ap.parse_args()
    turns = [r for r in map(json.loads, open(a.traj)) if r["episode_id"] == a.episode and r["agent"] == a.agent]
    turns.sort(key=lambda r: r["turn"])
    if a.max_turns:
        turns = turns[: a.max_turns]
    chain = turns[-1]["prompt_token_ids"]
    for t in turns:
        assert chain[: len(t["prompt_token_ids"])] == t["prompt_token_ids"], "history is not append-only"
    n_gen = sum(len(t["token_ids"]) for t in turns)
    per_turn_tokens = sum(len(t["prompt_token_ids"]) + len(t["token_ids"]) for t in turns)
    print(f"{len(turns)} turns, chain {len(chain)}, generated {n_gen}; packed {len(chain) + n_gen} tokens vs "
          f"per-turn {per_turn_tokens}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    torch.manual_seed(0)
    model = get_peft_model(base, LoraConfig(r=32, lora_alpha=64, target_modules=TARGETS, init_lora_weights=False))
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.mul_(0.02)
    model.train()
    tc = model.config.get_text_config()
    softcap, window = tc.final_logit_softcapping, tc.sliding_window
    weights = (torch.ones(n_gen) if a.weights == "ones"
               else torch.randn(n_gen, generator=torch.Generator().manual_seed(1))).cuda()  # stand-in advantages
    vllm_lp = torch.tensor([x for t in turns for x in t["logprobs"]])

    # per turn: one sample each, logits only for the generated tokens
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0, lps, off = time.time(), [], 0
    for t in turns:
        p, g = t["prompt_token_ids"], t["token_ids"]
        ids = torch.tensor([p + g], device="cuda")
        logits = model(input_ids=ids, logits_to_keep=len(g) + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[0, -len(g):, None])[:, 0]
        (lp * weights[off: off + len(g)]).sum().backward()
        lps.append(lp.detach())
        off += len(g)
    torch.cuda.synchronize()
    t_turn, m_turn = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    lp_turn, g_turn = torch.cat(lps).cpu(), grads(model)
    model.zero_grad(set_to_none=True)

    # noise floor: the same per-turn samples, backward in reverse turn order and in fp32 logits accumulation order
    offs = [0]
    for t in turns:
        offs.append(offs[-1] + len(t["token_ids"]))
    for i in reversed(range(len(turns))):
        t = turns[i]
        p, g = t["prompt_token_ids"], t["token_ids"]
        ids = torch.tensor([p + g], device="cuda")
        logits = model(input_ids=ids, logits_to_keep=len(g) + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, ids[0, -len(g):, None])[:, 0]
        (lp * weights[offs[i]: offs[i + 1]]).sum().backward()
    g_turn2 = grads(model)
    model.zero_grad(set_to_none=True)

    # packed: the chain once, each turn's tokens as a segment
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    pack = build_pack(torch.tensor(chain), [(len(t["prompt_token_ids"]), torch.tensor(t["token_ids"])) for t in turns],
                      window, "cuda")
    with active_pack(pack):
        lp, _ = packed_logps(model, pack, softcap, 1.0, tc.output_multiplier)
        (lp * weights).sum().backward()
    torch.cuda.synchronize()
    t_pack, m_pack = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    lp_pack, g_pack = lp.detach().cpu(), grads(model)

    d = (lp_pack - lp_turn).abs()
    num = sum(((g_pack[k] - g_turn[k]) ** 2).sum() for k in g_turn).sqrt()
    den = sum((g_turn[k] ** 2).sum() for k in g_turn).sqrt()
    cos = sum((g_pack[k] * g_turn[k]).sum() for k in g_turn) / (den * sum((g_pack[k] ** 2).sum() for k in g_pack).sqrt())
    def rel(x, y):
        return (sum(((x[k] - y[k]) ** 2).sum() for k in y).sqrt() / sum((y[k] ** 2).sum() for k in y).sqrt()).item()

    out = {"weights": a.weights, "turns": len(turns), "chain": len(chain), "generated": n_gen,
           "logp_pack_vs_turn": {"mean_abs": d.mean().item(), "max_abs": d.max().item()},
           "logp_turn_vs_vllm_mean_abs": (lp_turn - vllm_lp).abs().mean().item(),
           "logp_pack_vs_vllm_mean_abs": (lp_pack - vllm_lp).abs().mean().item(),
           "grad_rel_err": (num / den).item(), "grad_cos": cos.item(),
           "grad_rel_err_noise_floor(per-turn reordered)": rel(g_turn2, g_turn), "grad_tensors": len(g_turn),
           "per_turn": {"s": t_turn, "peak_gib": m_turn}, "packed": {"s": t_pack, "peak_gib": m_pack}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
