"""GRPO steps on a finished groups run (scripts/partnr_groups.sh), on one GPU: checks the training side end to end
(groups, advantages, packing, loss, LoRA update) and measures step time and memory. Each step takes the next
<groups-per-step> groups; the adapter is saved after every step (<out>/step_<n>).

    envs/trl/bin/python -I scripts/grpo_offline.py runs/<groups run> <out> [--groups-per-step 1] [--steps 3]
"""
import argparse
import json
import time
from pathlib import Path

import torch

from muse_habitat.grpo import groups, load_rollouts, packs, train_step
from muse_habitat.models import GLIMMER as MODEL
# all text layers; not the attention output gate (vLLM may map its name into the MLP's packed gate_up_proj)
TARGETS = r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--groups-per-step", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--reward", default="success")
    ap.add_argument("--budget", type=int, default=12288, help="max tokens per packed sequence")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ros = load_rollouts(a.run, a.reward)
    gs = groups(ros)
    n_ro = len({k[1] for k in ros})
    print(f"{len(ros)} rollouts of {n_ro} episodes; {len(gs)} groups with reward spread "
          f"(mean reward {sum(r.reward for r in ros.values()) / len(ros):.3f})", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=TARGETS))
    model.train()
    tc = model.config.get_text_config()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.99),
                            weight_decay=0.0)
    log = open(out / "log.jsonl", "a")
    for step in range(a.steps):
        batch = gs[step * a.groups_per_step:(step + 1) * a.groups_per_step]
        if not batch:
            break
        pl = packs(batch, a.budget)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        st = train_step(model, opt, pl, tc)
        torch.cuda.synchronize()
        st.update(step=step, seconds=time.time() - t0, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                  episodes=[g[0].episode_id for g in batch], rollouts=sum(len(g) for g in batch),
                  packed_tokens=sum(p.n_tokens for p in pl), max_pack=max(p.n_tokens for p in pl),
                  rewards=[[r.reward for r in g] for g in batch])
        print(json.dumps(st), flush=True)
        log.write(json.dumps(st) + "\n")
        log.flush()
        model.save_pretrained(out / f"step_{step}")


if __name__ == "__main__":
    main()
