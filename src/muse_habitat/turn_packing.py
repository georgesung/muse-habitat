"""Prefix-shared packing of a branch's per-turn training samples.

With thinking stripped from the history, each policy turn of a multi-turn episode is its own
sample: the context it was generated from plus its thinking and reply. Turn k+1's context
extends turn k's, so a branch packs into one sequence

    [ chain | seg_0 | seg_1 | ... ]

where chain is the last turn's context and seg_k turn k's completion. seg_k takes positions
h_k.. (h_k = length of turn k's context) and sees chain[:h_k] plus itself causally, which is
exactly the context it was sampled with. The history is then computed once per branch instead
of once per turn.
"""

from contextlib import contextmanager

import torch


def _ranges(chain_len, seg_ctx, seg_len, window, device):
    """Two key ranges per query row and two query ranges per key row (see ranged_attn)."""
    seg_ctx = torch.as_tensor(seg_ctx, dtype=torch.long)
    seg_len = torch.as_tensor(seg_len, dtype=torch.long)
    n_seg = len(seg_len)
    seg_start = chain_len + torch.cat(
        [torch.zeros(1, dtype=torch.long), seg_len.cumsum(0)[:-1]]
    )
    n = chain_len + int(seg_len.sum())
    big = n
    w = window if window else n + 1

    idx = torch.arange(n)
    is_chain = idx < chain_len
    seg_id = torch.zeros(n, dtype=torch.long)
    if n_seg:
        seg_id[chain_len:] = torch.repeat_interleave(torch.arange(n_seg), seg_len)
    s_start = seg_start[seg_id]
    s_ctx = seg_ctx[seg_id]
    s_end = s_start + seg_len[seg_id] - 1
    pos = torch.where(is_chain, idx, s_ctx + (idx - s_start))

    # Query side. Chain rows: B = causal (windowed). Segment rows: A = context prefix, B = own segment.
    qa_lo = torch.where(is_chain, big, (pos - w + 1).clamp(min=0))
    qa_hi = torch.where(is_chain, -1, s_ctx - 1)
    a_empty = qa_lo > qa_hi
    qa_lo = torch.where(a_empty, big, qa_lo)
    qa_hi = torch.where(a_empty, -1, qa_hi)
    qb_lo = torch.where(
        is_chain, (idx - w + 1).clamp(min=0), torch.maximum(s_start, idx - w + 1)
    )
    qb_hi = idx

    # Key side (may be a superset). Chain key j: chain rows [j, j+w-1], plus segment rows of the
    # turns whose context covers j, within the window: segments k1..k2.
    kb_lo = idx.clone()
    kb_hi = torch.where(
        is_chain,
        torch.clamp(idx + w - 1, max=chain_len - 1),
        torch.minimum(s_end, idx + w - 1),
    )
    ka_lo = torch.full((n,), big, dtype=torch.long)
    ka_hi = torch.full((n,), -1, dtype=torch.long)
    if n_seg:
        j = idx[:chain_len]
        k1 = torch.searchsorted(seg_ctx, j, right=True)
        k2 = torch.searchsorted(seg_ctx, j + w - 1, right=True) - 1
        ok = k1 <= k2
        k2c = k2.clamp(min=0)
        hi = seg_start[k2c] + torch.minimum(seg_len[k2c] - 1, j + w - 1 - seg_ctx[k2c])
        ka_lo[:chain_len] = torch.where(ok, seg_start[k1.clamp(max=n_seg - 1)], big)
        ka_hi[:chain_len] = torch.where(ok, hi, -1)
    out = [qa_lo, qa_hi, qb_lo, qb_hi, ka_lo, ka_hi, kb_lo, kb_hi]
    return pos, [t.to(device=device, dtype=torch.int32) for t in out]


def build_pack(chain_ids, segments, sliding_window, device):
    """chain_ids: 1-D LongTensor; segments: list of (context_len, completion_ids) in turn order.

    Returns input_ids [1, N], position_ids [1, N], the attention ranges for sliding and global
    layers, and for every completion token the row of the hidden state that predicts it."""
    seg_ctx = [c for c, _ in segments]
    seg_len = [len(ids) for _, ids in segments]
    if any(b <= a for a, b in zip(seg_ctx, seg_ctx[1:], strict=False)):
        raise ValueError("turn contexts must grow strictly")
    if seg_ctx and seg_ctx[-1] > len(chain_ids):
        raise ValueError("a turn context is longer than the chain")
    chain_len = len(chain_ids)
    input_ids = torch.cat(
        [chain_ids.to(device)] + [ids.to(device) for _, ids in segments]
    )
    pos, sliding = _ranges(chain_len, seg_ctx, seg_len, sliding_window, device)
    _, full = _ranges(chain_len, seg_ctx, seg_len, None, device)
    # Token 0 of seg_k is predicted from chain[h_k - 1], token r>0 from seg_k[r-1].
    pred_rows = []
    start = chain_len
    for ctx_len, n in zip(seg_ctx, seg_len, strict=True):
        pred_rows.append(torch.tensor([ctx_len - 1]))
        pred_rows.append(torch.arange(start, start + n - 1))
        start += n
    pred = torch.cat(pred_rows) if pred_rows else torch.zeros(0, dtype=torch.long)
    return {
        "input_ids": input_ids[None],
        "position_ids": pos.to(device)[None],
        "ranges_sliding": sliding,
        "ranges_full": full,
        "pred_rows": pred.to(device),
        "seg_len": seg_len,
    }


# ---------------------------------------------------------------------------------------
# Attention routing
# ---------------------------------------------------------------------------------------

_ACTIVE: dict | None = None
_ORIG_ATTR = "_muse_turn_pack_orig"


def _packed_attention(module, query, key, value, attention_mask, **kwargs):
    from muse_habitat.ranged_attn import ranged_attention

    pack = _ACTIVE
    window = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
    ranges = pack["ranges_sliding"] if window else pack["ranges_full"]
    scaling = kwargs.get("scaling")
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    out = ranged_attention(query[0], key[0], value[0], ranges, scaling)
    return out.transpose(0, 1).unsqueeze(0), None


def install_packed_attention(names=("flash_attention_2", "sdpa", "sdpa_global_packed")):
    """Wrap the named attention functions so they take the packed path while a pack is active."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for name in names:
        if name not in ALL_ATTENTION_FUNCTIONS.valid_keys():
            continue
        current = ALL_ATTENTION_FUNCTIONS[name]
        if getattr(current, _ORIG_ATTR, None) is not None:
            continue

        def wrapper(
            module, query, key, value, attention_mask, *args, _orig=current, **kwargs
        ):
            if _ACTIVE is not None:
                return _packed_attention(
                    module, query, key, value, attention_mask, **kwargs
                )
            return _orig(module, query, key, value, attention_mask, *args, **kwargs)

        setattr(wrapper, _ORIG_ATTR, current)
        ALL_ATTENTION_FUNCTIONS.register(name, wrapper)


@contextmanager
def active_pack(pack):
    """Route attention through the pack's ranges. Must stay active through the backward when
    gradient checkpointing recomputes attention."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, pack
    try:
        yield
    finally:
        _ACTIVE = prev


def _lm_head_pre_hook(module, args):
    """During a packed forward: log-probs of the packed completion tokens, chunk by chunk,
    from lm_head's input rows; lm_head itself then projects a single row."""
    pack = _ACTIVE
    if pack is None or "want_logps" not in pack:
        return None
    targets, softcap, temperature, scale = pack.pop("want_logps")
    # FSDP frees the unsharded weight after the forward; the backward recomputes logits.
    weight = module.weight.detach().clone()
    pack["logps"] = chunked_logps(args[0][0], weight, targets, softcap, temperature, scale)
    return (args[0][:, :1],)


def packed_logps(model, pack, softcap, temperature, scale=1.0):
    """Forward one pack (inside active_pack) and return the log-prob and entropy of every
    completion token, segments in order. The model must take position_ids, a per-layer-type
    attention_mask dict and a tensor logits_to_keep (Muse Glimmer does). scale
    multiplies the logits before the softcap (Muse Glimmer's output_multiplier)."""
    install_packed_attention()
    lm_head = next(m for n, m in model.named_modules() if n.endswith("lm_head"))
    if not getattr(lm_head, "_muse_pack_hook", False):
        lm_head.register_forward_pre_hook(_lm_head_pre_hook)
        lm_head._muse_pack_hook = True
    chain_len = pack["input_ids"].size(1) - sum(pack["seg_len"])
    pack["want_logps"] = (pack["input_ids"][0, chain_len:], softcap, temperature, scale)
    out = model(
        input_ids=pack["input_ids"],
        position_ids=pack["position_ids"],
        attention_mask={"full_attention": None, "sliding_attention": None},
        logits_to_keep=pack["pred_rows"],
        use_cache=False,
    )
    logps, ent = pack.pop("logps")
    if logps.requires_grad:
        # FSDP re-gathers the root's params (final norm, lm_head) for the backward from a
        # hook on the model's outputs, so the gradient must also flow through them.
        logps = logps + out.logits.sum().to(logps.dtype) * 0.0
    return logps, ent


# ---------------------------------------------------------------------------------------
# Log-probs of selected rows without materializing all logits
# ---------------------------------------------------------------------------------------


def _chunk_logits(h, weight, softcap, temperature, scale):
    z = (h @ weight.t()).float() * scale
    if softcap:
        t = torch.tanh(z / softcap)
        return t * softcap / temperature, t
    return z / temperature, None


class _ChunkedLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, targets, softcap, temperature, scale, chunk):
        logps = torch.empty(h.shape[0], device=h.device, dtype=torch.float32)
        ent = torch.empty_like(logps)
        for s in range(0, h.shape[0], chunk):
            logits, _ = _chunk_logits(
                h[s : s + chunk], weight, softcap, temperature, scale
            )
            lsm = torch.log_softmax(logits, -1)
            logps[s : s + chunk] = lsm.gather(1, targets[s : s + chunk, None])[:, 0]
            ent[s : s + chunk] = -(lsm.exp() * lsm).sum(-1)
        ctx.save_for_backward(h, weight, targets)
        ctx.softcap, ctx.temperature, ctx.scale, ctx.chunk = (
            softcap,
            temperature,
            scale,
            chunk,
        )
        ctx.mark_non_differentiable(ent)
        return logps, ent

    @staticmethod
    def backward(ctx, g_logps, _g_ent):
        h, weight, targets = ctx.saved_tensors
        grad_h = torch.empty_like(h)
        for s in range(0, h.shape[0], ctx.chunk):
            logits, t = _chunk_logits(
                h[s : s + ctx.chunk], weight, ctx.softcap, ctx.temperature, ctx.scale
            )
            g = g_logps[s : s + ctx.chunk, None].float()
            grad = -torch.softmax(logits, -1) * g
            grad.scatter_add_(1, targets[s : s + ctx.chunk, None], g)
            grad = grad * (ctx.scale / ctx.temperature)
            if t is not None:
                grad = grad * (1 - t * t)
            grad_h[s : s + ctx.chunk] = (grad.to(weight.dtype) @ weight).to(h.dtype)
        return grad_h, None, None, None, None, None, None


def chunked_logps(h, weight, targets, softcap, temperature, scale=1.0, chunk=1024):
    """Log-prob of each target and the entropy (no grad) of softcap(scale * h @ W^T) / temperature,
    one chunk of rows at a time; the backward recomputes each chunk's logits. W gets no grad."""
    return _ChunkedLogps.apply(
        h, weight.detach(), targets, softcap, temperature, scale, chunk
    )
