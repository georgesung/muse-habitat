"""Packed log-probs and ranged attention against plain PyTorch autograd (needs a GPU and Triton).

    envs/trl/bin/python -I -m pytest tests/test_turn_packing.py
"""
import pytest
import torch

from muse_habitat.ranged_attn import ranged_attention
from muse_habitat.turn_packing import _ranges, chunked_logps

dev = "cuda"


def test_chunked_logps_scale_softcap():
    torch.manual_seed(0)
    h = torch.randn(70, 64, device=dev, dtype=torch.float64, requires_grad=True)
    w = torch.randn(300, 64, device=dev, dtype=torch.float64)
    tgt = torch.randint(0, 300, (70,), device=dev)
    g = torch.randn(70, device=dev, dtype=torch.float64)
    scale, cap, temp = 0.196, 20.0, 0.9
    ref = torch.log_softmax(cap * torch.tanh(h @ w.t() * scale / cap) / temp, -1).gather(1, tgt[:, None])[:, 0]
    (gr,) = torch.autograd.grad((ref * g).sum(), h)
    lp, _ = chunked_logps(h, w, tgt, cap, temp, scale, chunk=32)
    (gp,) = torch.autograd.grad((lp * g).sum(), h)
    assert torch.allclose(lp.double(), ref, atol=1e-5)
    assert torch.allclose(gp, gr, rtol=1e-4, atol=1e-6)


def reference_mask(chain_len, seg_ctx, seg_len, window):
    """Dense mask and positions of the packed layout, from its definition."""
    pos, rows = list(range(chain_len)), []
    for c, n in zip(seg_ctx, seg_len):
        pos += list(range(c, c + n))
    n = len(pos)
    m = torch.zeros(n, n, dtype=torch.bool)
    owner = [-1] * chain_len + [k for k, s in enumerate(seg_len) for _ in range(s)]
    for i in range(n):
        for j in range(i + 1):
            if owner[j] == -1:
                ok = j < (chain_len if owner[i] == -1 else seg_ctx[owner[i]]) and (owner[i] != -1 or j <= i)
            else:
                ok = owner[j] == owner[i]
            if ok and window:
                ok = pos[i] - pos[j] < window
            m[i, j] = ok
    return m, torch.tensor(pos)


@pytest.mark.parametrize("window", [None, 48])
@pytest.mark.parametrize("hq,hk,d", [(4, 2, 64), (32, 2, 128)])
def test_ranged_attention_vs_dense(window, hq, hk, d):
    torch.manual_seed(0)
    chain_len, seg_ctx, seg_len = 150, [40, 90, 150], [23, 31, 17]
    n = chain_len + sum(seg_len)
    mask, _ = reference_mask(chain_len, seg_ctx, seg_len, window)
    _, ranges = _ranges(chain_len, seg_ctx, seg_len, window, dev)
    q = torch.randn(n, hq, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(n, hk, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(n, hk, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    go = torch.randn(n, hq, d, device=dev, dtype=torch.float32)
    s = d ** -0.5
    # transformers layout: [B, H, N, D]
    out = ranged_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), ranges, s)
    gq, gk, gv = torch.autograd.grad((out.float() * go.transpose(0, 1)).sum(), (q, k, v))
    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    kr, vr = (x.repeat_interleave(hq // hk, 1) for x in (kf, vf))
    att = torch.einsum("qhd,khd->hqk", qf, kr) * s
    att = att.masked_fill(~mask.to(dev)[None], float("-inf")).softmax(-1)
    ref = torch.einsum("hqk,khd->hqd", att, vr)
    rq, rk, rv = torch.autograd.grad((ref * go.transpose(0, 1)).sum(), (qf, kf, vf))
    assert torch.allclose(out.float(), ref, atol=2e-2), (out.float() - ref).abs().max()
    for a, b, name in ((gq, rq, "dq"), (gk, rk, "dk"), (gv, rv, "dv")):
        err = (a.float() - b).norm() / b.norm()
        assert err < 2e-2, f"{name} rel err {err:.3f}"
