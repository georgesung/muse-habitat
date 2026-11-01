"""Memory and time of one LoRA policy-gradient step on Glimmer 30B on one GPU (transformers + PEFT, bf16 base,
gradient checkpointing), for sequence lengths a GRPO trainer on PARTNR turns would see.

Each step: one sequence of <len> tokens (random ids: cost does not depend on content), the loss on its last
<completion> tokens only (logits computed just there), backward, AdamW on the LoRA parameters.

    envs/trl/bin/python -I scripts/train_step_bench.py [--lens 4096,8192,16384] [--completion 512] [--rank 32]
"""
import argparse
import json
import time

import torch

MODEL = ("/home/user/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/"
         "cec94fcfd6512f2cfe708138c32533b346ea4f69")
TARGETS = r".*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="4096,8192,16384")
    ap.add_argument("--completion", type=int, default=512)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText
    base = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, target_modules=TARGETS))
    model.train()  # transformers checkpoints layers only in training mode; from_pretrained returns eval mode
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    vocab = model.config.get_text_config().vocab_size
    print(f"weights {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    out = []
    for n in map(int, a.lens.split(",")):
        times = []
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(a.steps):
                ids = torch.randint(1000, vocab - 1000, (1, n), device="cuda")
                torch.cuda.synchronize()
                t0 = time.time()
                logits = model(input_ids=ids, logits_to_keep=a.completion + 1).logits[0, :-1].float()
                logp = torch.log_softmax(logits, -1).gather(-1, ids[0, -a.completion:, None])[:, 0]
                loss = -(logp * torch.randn_like(logp)).mean()  # advantage-weighted log-likelihood
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            r = {"tokens": n, "step_s": min(times), "tok_per_s": n / min(times),
                 "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        except torch.OutOfMemoryError:
            r = {"tokens": n, "oom": True}
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        print(r, flush=True)
        out.append(r)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
