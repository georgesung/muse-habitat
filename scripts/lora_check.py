"""LoRA round trip between the trainer (transformers + PEFT) and vLLM, on real PARTNR turns.

1. Build a seeded test adapter on Glimmer's text model (non-zero, so it changes outputs) and save it.
2. Load it into a running vLLM server (--enable-lora, runtime loading) under <name>.
3. For each logged turn (trajectory JSONL from MUSE_TRAJ_DIR runs): sample a completion at temperature 1 from vLLM,
   with the base model and with the adapter, keeping the sampled tokens' logprobs.
4. Score the same tokens with the trainer-side forward (base: adapter disabled) and compare per-token logprobs.

    envs/trl/bin/python -I scripts/lora_check.py <traj jsonl> <out dir> [--port 8021] [--turns 8] [--gpu 0]
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import torch
from muse_habitat.models import GLIMMER as MODEL
# text model only, without the attention output gate (self_attn.gate_proj): vLLM maps ".gate_proj" to the MLP's
# packed gate_up_proj as well as renaming the attention gate, so an adapter on it may load into the wrong module
TARGETS = r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def sample(port, model, prompt_ids, max_tokens, seed):
    out = json.loads(post(f"http://127.0.0.1:{port}/v1/completions", {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 1.0, "seed": seed,
        "logprobs": 0, "return_token_ids": True, "skip_special_tokens": False}))["choices"][0]
    return out["token_ids"], out["logprobs"]["token_logprobs"]


@torch.no_grad()
def score(model, prompt_ids, gen_ids):
    ids = torch.tensor([prompt_ids + gen_ids], device=model.device)
    logits = model(input_ids=ids, logits_to_keep=len(gen_ids) + 1).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(gen_ids, device=logits.device)[:, None])[:, 0].cpu()


def stats(d):
    d = d.abs()
    return {"tokens": len(d), "mean_abs": d.mean().item(), "p99_abs": d.quantile(0.99).item(), "max_abs": d.max().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--b-scale", type=float, default=0.02, help="scale of the random lora_B (adapter strength)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in open(a.traj)]
    recs = [r for r in recs if r.get("prompt_token_ids")]
    random.Random(0).shuffle(recs)
    recs = recs[: a.turns]
    print(f"{len(recs)} turns, prompt tokens {[len(r['prompt_token_ids']) for r in recs]}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    t0 = time.time()
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    print(f"loaded in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.manual_seed(0)
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=TARGETS,
                                            init_lora_weights=False))
    for n, p in model.named_parameters():  # random A and B, scaled so the adapter shifts but does not wreck outputs
        if "lora_B" in n:
            p.data.mul_(a.b_scale)
    model.print_trainable_parameters()
    adapter = out / "adapter"
    model.save_pretrained(adapter)

    try:  # a previous run's adapter under the same name
        post(f"http://127.0.0.1:{a.port}/v1/unload_lora_adapter", {"lora_name": "test"})
    except Exception:
        pass
    post(f"http://127.0.0.1:{a.port}/v1/load_lora_adapter", {"lora_name": "test", "lora_path": str(adapter)})
    print("adapter loaded into vLLM", flush=True)

    results = {"base": [], "lora": [], "shift": []}
    for i, r in enumerate(recs):
        for name, served in (("base", "glimmer"), ("lora", "test")):
            gen, lp_vllm = sample(a.port, served, r["prompt_token_ids"], a.max_tokens, seed=i)
            if name == "base":
                with model.disable_adapter():
                    lp_hf = score(model, r["prompt_token_ids"], gen)
            else:
                lp_hf = score(model, r["prompt_token_ids"], gen)
                with model.disable_adapter():  # how much the adapter moves the policy, on its own samples
                    results["shift"].append(lp_hf - score(model, r["prompt_token_ids"], gen))
            d = lp_hf - torch.tensor(lp_vllm)
            results[name].append(d)
            print(f"turn {i} ({len(r['prompt_token_ids'])} prompt) {name}: {stats(d)} "
                  f"mean logprob vllm {sum(lp_vllm) / len(lp_vllm):.3f}", flush=True)
    summary = {k: stats(torch.cat(v)) for k, v in results.items()}
    # base / lora: trainer minus vLLM logprob of the sampled tokens; shift: trainer with minus without the adapter
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
