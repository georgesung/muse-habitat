"""True gradients of Glimmer 30B in fp32 on one GPU, layer by layer, and how bf16 attention kernels compare.

The model stays on the CPU in bf16. fp32 truth: each decoder layer is copied to the GPU in fp32 (the real bf16 weights,
upcast), the forward keeps only each layer's input, and the backward goes from the top layer down, recomputing one
layer at a time. Loss: sum of the log-probs of the generated tokens (softcapped head, as the model computes it).

For each bf16 attention implementation (sdpa, eager, the ranged Triton kernel):
- chained: the whole forward and backward in bf16, as training sees it;
- local: each layer gets the fp32 input and fp32 upstream gradient, so only that layer's backward is bf16.
Per layer: relative error of dL/d(layer input) and of the o_proj / down_proj weight gradients vs fp32.

    envs/trl/bin/python -I scripts/grad_truth.py <turns.json> [--turns 2] [--out runs/grad_truth]
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

MODEL = ("/home/user/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/"
         "cec94fcfd6512f2cfe708138c32533b346ea4f69")
WEIGHTS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


class Text:
    """Glimmer's text model run one layer at a time on the GPU in a given dtype and attention implementation."""

    def __init__(self, model, dtype, impl, autocast=False, fp32_parts=()):
        """autocast: bf16 weights and matmuls under torch.autocast, fp32 residual stream (dtype is then fp32)."""
        self.lm = model.model.language_model
        self.cfg = self.lm.config
        self.head = model.lm_head
        self.dtype, self.impl, self.autocast = (torch.float32 if autocast else dtype), impl, autocast
        self.fp32_parts = set(fp32_parts)  # parts of each bf16 layer run in fp32: attn, core, mlp, norms

    def ctx(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast)

    def gpu(self, module):
        m = copy.deepcopy(module).to("cuda", torch.bfloat16 if self.autocast else self.dtype)
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    def setup(self, ids, n_gen):
        self.ids, self.n_gen = ids, n_gen
        n = ids.shape[1]
        self.pos = torch.arange(n, device="cuda")[None]
        h = self.gpu(self.lm.embed_tokens)(ids).to(self.dtype)
        self.cfg._attn_implementation = "sdpa" if self.impl == "kernel" else self.impl
        kw = {"config": self.cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
              "position_ids": self.pos}
        self.masks = {"full_attention": create_causal_mask(**kw),
                      "sliding_attention": create_sliding_window_causal_mask(**kw)}
        with self.ctx():
            self.rope = self.gpu(self.lm.rotary_emb)(h, self.pos)
        self.pack = None
        if self.impl == "kernel":
            from muse_habitat.turn_packing import build_pack, install_packed_attention
            install_packed_attention()
            p = ids[0, :n - n_gen].cpu()
            self.pack = build_pack(p, [(len(p), ids[0, n - n_gen:].cpu())], self.cfg.sliding_window, "cuda")
        return h

    def layer(self, i, h, weight_grads=False):
        """Run layer i on h (a leaf); returns (out, layer module)."""
        from muse_habitat.turn_packing import active_pack
        layer = self.gpu(self.lm.layers[i])
        upcast(layer, self.fp32_parts)
        for n, p in layer.named_parameters():
            p.requires_grad_(weight_grads and n in WEIGHTS)
        rope = self.rope if self.cfg.layer_rope_theta[i] else None
        with self.ctx(), core32("core" in self.fp32_parts):
            if self.pack is not None:
                with active_pack(self.pack):
                    out = layer(h, attention_mask=None, position_embeddings=rope, position_ids=self.pos)
            else:
                out = layer(h, attention_mask=self.masks[self.cfg.layer_types[i]], position_embeddings=rope,
                            position_ids=self.pos)
        return out.to(self.dtype), layer

    def backward_layer(self, i, h, g):
        """Recompute layer i from h and backpropagate g: (dL/dh, {weight: grad})."""
        from muse_habitat.turn_packing import active_pack
        h = h.detach().to(self.dtype).requires_grad_()
        out, layer = self.layer(i, h, weight_grads=True)
        with core32("core" in self.fp32_parts):
            if self.pack is not None:
                with active_pack(self.pack):
                    out.backward(g.to(self.dtype))
            else:
                out.backward(g.to(self.dtype))
        wg = {n: p.grad.float() for n, p in layer.named_parameters() if p.grad is not None}
        return h.grad.float(), wg

    def head_grad(self, h_last):
        """Loss and dL/d(input of the final norm), the loss in fp32 from the head's output in self.dtype."""
        h = h_last.detach().to(self.dtype).requires_grad_()
        norm, head = self.gpu(self.lm.norm), self.gpu(self.head)
        n = self.ids.shape[1]
        with self.ctx():
            logits = head(norm(h)[0, n - self.n_gen - 1:n - 1]).float()
        tc = self.cfg
        logits = torch.tanh(logits * tc.output_multiplier / tc.final_logit_softcapping) * tc.final_logit_softcapping
        lp = torch.log_softmax(logits, -1).gather(-1, self.ids[0, n - self.n_gen:, None])[:, 0]
        lp.sum().backward()
        return lp.detach(), h.grad.float()

    @torch.no_grad()
    def forward_states(self, h):
        states = [h]
        for i in range(self.cfg.num_hidden_layers):
            h, _ = self.layer(i, h)
            states.append(h)
        return states


def _to32(m):
    """Run module m in fp32: inputs upcast, outputs cast back to the caller's dtype."""
    m.float()

    def pre(mod, args, kwargs):
        mod._dt = next((a.dtype for a in args if torch.is_tensor(a)), torch.bfloat16)
        cast = lambda x: x.float() if torch.is_tensor(x) and x.is_floating_point() else x
        return tuple(map(cast, args)), {k: (tuple(map(cast, v)) if isinstance(v, tuple) else cast(v))
                                        for k, v in kwargs.items()}

    def post(mod, args, kwargs, out):
        back = lambda x: x.to(mod._dt) if torch.is_tensor(x) and x.is_floating_point() else x
        return tuple(map(back, out)) if isinstance(out, tuple) else back(out)
    m.register_forward_pre_hook(pre, with_kwargs=True)
    m.register_forward_hook(post, with_kwargs=True)


def upcast(layer, parts):
    if "attn" in parts:
        _to32(layer.self_attn)
    if "mlp" in parts:
        _to32(layer.mlp)
    if "norms" in parts:
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            _to32(getattr(layer, n))


class core32:
    """While active, sdpa computes attention (scores, softmax, weighted sum) in fp32."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        if self.on:
            from transformers.integrations.sdpa_attention import sdpa_attention_forward
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            self.prev = ALL_ATTENTION_FUNCTIONS["sdpa"]

            def f(module, q, k, v, mask, **kw):
                out, w = sdpa_attention_forward(module, q.float(), k.float(), v.float(), mask, **kw)
                return out.to(q.dtype), w
            ALL_ATTENTION_FUNCTIONS["sdpa"] = f

    def __exit__(self, *exc):
        if self.on:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["sdpa"] = self.prev


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("turns")
    ap.add_argument("--turns-used", type=int, default=2)
    ap.add_argument("--out", default="runs/grad_truth")
    # <forward>><backward>, each <impl>[:fp32|:ac]; forward "fp32" = the truth's states, "local" = per-layer
    ap.add_argument("--configs", default="sdpa,fp32>sdpa,sdpa>sdpa:fp32,sdpa:ac,kernel:ac,local>sdpa:ac")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    turns = json.load(open(a.turns))[: a.turns_used]

    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
    L = model.config.get_text_config().num_hidden_layers
    results = []
    for ti, t in enumerate(turns):
        ids = torch.tensor([t["prompt_token_ids"] + t["token_ids"]], device="cuda")
        n_gen = len(t["token_ids"])
        # fp32 truth
        tr = Text(model, torch.float32, "eager")
        states = tr.forward_states(tr.setup(ids, n_gen))
        lp32, g = tr.head_grad(states[-1])
        gin32, wg32 = [None] * (L + 1), [None] * L
        gin32[L] = g
        for i in reversed(range(L)):
            g, wg32[i] = tr.backward_layer(i, states[i], g)
            gin32[i] = g
        print(f"turn {ti}: {ids.shape[1]} tokens, {n_gen} generated, fp32 mean logp {lp32.mean():.4f}; "
              f"|dL/dh| top {gin32[L].norm():.3e} layer 0 {gin32[0].norm():.3e}", flush=True)
        # each config: which Text makes the forward states, which does the backward; "local" feeds fp32 states and
        # fp32 upstream gradients into each layer's backward
        def T(spec):  # <impl>[:fp32|:ac][+part...]
            spec, *parts = spec.split("+")
            impl, _, kind = spec.partition(":")
            return Text(model, torch.float32 if kind == "fp32" else torch.bfloat16, impl, autocast=kind == "ac",
                        fp32_parts=parts)
        for spec in a.configs.split(","):
            fwd, _, bwd = spec.partition(">")
            bwd = bwd or fwd
            bt = T(bwd)
            h0 = bt.setup(ids, n_gen)
            if fwd == "local":
                st, mode = states, "local"
            elif fwd == "fp32":
                st, mode = states, "chained"
            else:
                ft = T(fwd)
                st, mode = ft.forward_states(ft.setup(ids, n_gen)), "chained"
                bt.setup(ids, n_gen)
            lp, g = bt.head_grad(st[-1])
            rows = []
            for i in reversed(range(L)):
                up = gin32[i + 1] if mode == "local" else g
                g, wg = bt.backward_layer(i, st[i], up)
                rows.append({"layer": i, "dh": rel(g, gin32[i]),
                             **{w.split(".")[1]: rel(wg[w], wg32[i][w]) for w in WEIGHTS}})
            rows.reverse()
            r = {"turn": ti, "config": spec, "rows": rows,
                 "logp_vs_fp32": (lp - lp32).abs().mean().item()}
            results.append(r)
            show = [0, 4, 8, 12, 16, 20, 24, 30, 40, 51]
            print(f"  {spec:24s} logp|d| {r['logp_vs_fp32']:.4f}  dL/dh rel err: "
                  + " ".join(f"{x['layer']}:{x['dh']:.3f}" for x in rows if x["layer"] in show), flush=True)
            print(f"  {'':24s} down_proj grad rel err: "
                  + " ".join(f"{x['layer']}:{x['down_proj']:.3f}" for x in rows if x["layer"] in show), flush=True)
        json.dump(results, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
