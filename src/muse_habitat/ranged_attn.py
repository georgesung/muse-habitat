"""Triton flash attention where each query row sees up to two key ranges of the same sequence.

Query row i attends keys j in [qa_lo[i], qa_hi[i]] or [qb_lo[i], qb_hi[i]] (an empty range has
lo > hi). That expresses prefix-shared packing of multi-turn samples: one sequence holds a branch's
history once, followed by each turn's generated segment; a segment row sees the history prefix the
turn was generated from (range A) and its own segment causally (range B), a history row sees the
history causally (range B). A sliding window is folded into the ranges by the caller.

The backward loops over the query rows that can see each key block, given as two ranges per key
row as well (ka_*, kb_*); they may be supersets, the exact mask always comes from the query side.

GQA in-kernel (kv head = q head // groups), so K/V are never repeated. B=1 only.
"""

import torch
import triton
import triton.language as tl


def _smem_limit() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (
        getattr(props, "shared_memory_per_block_optin", 0)
        or props.shared_memory_per_block
    )


def _tiles(head_dim: int) -> tuple[dict, dict, dict]:
    """(forward, dK/dV, dQ) launch configs."""
    big = _smem_limit() >= 200 * 1024
    if head_dim > 256:
        if big:
            return (
                {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1},
                {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 8, "num_stages": 1},
                {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            )
        small = {"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1}
        return small, dict(small), dict(small)
    if big:
        return (
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 8, "num_stages": 1},
            {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
        )
    small = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1}
    return small, dict(small), dict(small)


def _prec(t: torch.Tensor) -> str:
    return "ieee" if t.dtype == torch.float32 else "tf32"


@triton.jit
def _in_ranges(cols, a_lo, a_hi, b_lo, b_hi):
    return ((cols[None, :] >= a_lo[:, None]) & (cols[None, :] <= a_hi[:, None])) | (
        (cols[None, :] >= b_lo[:, None]) & (cols[None, :] <= b_hi[:, None])
    )


@triton.jit
def _loop_bounds(lo_a, hi_a, lo_b, hi_b, BLOCK: tl.constexpr):
    """Block-aligned [start, end) of two passes that together visit every block once. The first
    range must lie before the second."""
    a_start = (tl.min(lo_a) // BLOCK) * BLOCK
    a_end = tl.max(hi_a) + 1
    a_stop = tl.where(
        a_end > a_start, a_start + tl.cdiv(a_end - a_start, BLOCK) * BLOCK, 0
    )
    b_start = tl.maximum((tl.min(lo_b) // BLOCK) * BLOCK, a_stop)
    b_end = tl.max(hi_b) + 1
    return a_start, a_end, b_start, b_end


@triton.jit
def _fwd_block(
    q,
    acc,
    m_i,
    l_i,
    start_n,
    kb,
    vb,
    skn,
    a_lo,
    a_hi,
    b_lo,
    b_hi,
    sm_scale,
    N,
    offs_n,
    offs_d,
    PREC: tl.constexpr,
):
    cols = start_n + offs_n
    k = tl.load(
        kb + cols[None, :] * skn + offs_d[:, None], mask=cols[None, :] < N, other=0.0
    )
    qk = tl.dot(q, k, input_precision=PREC) * sm_scale
    allowed = _in_ranges(cols, a_lo, a_hi, b_lo, b_hi) & (cols[None, :] < N)
    qk = tl.where(allowed, qk, -1.0e9)
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.where(allowed, tl.exp(qk - m_new[:, None]), 0.0)
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    v = tl.load(
        vb + cols[:, None] * skn + offs_d[None, :], mask=cols[:, None] < N, other=0.0
    )
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PREC)
    return acc, m_new, l_i


@triton.jit
def _fwd(
    Q,
    K,
    V,
    O,  # noqa: E741
    L,
    QA_LO,
    QA_HI,
    QB_LO,
    QB_HI,
    sm_scale,
    sqh,
    sqn,
    skh,
    skn,
    N,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // GROUPS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    rows = offs_m < N
    q = tl.load(
        Q + h * sqh + offs_m[:, None] * sqn + offs_d[None, :],
        mask=rows[:, None],
        other=0.0,
    )
    a_lo = tl.load(QA_LO + offs_m, mask=rows, other=N)
    a_hi = tl.load(QA_HI + offs_m, mask=rows, other=-1)
    b_lo = tl.load(QB_LO + offs_m, mask=rows, other=N)
    b_hi = tl.load(QB_HI + offs_m, mask=rows, other=-1)
    kb = K + hk * skh
    vb = V + hk * skh
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    a_start, a_end, b_start, b_end = _loop_bounds(a_lo, a_hi, b_lo, b_hi, BLOCK_N)
    for start_n in range(a_start, a_end, BLOCK_N):
        acc, m_i, l_i = _fwd_block(
            q, acc, m_i, l_i, start_n, kb, vb, skn, a_lo, a_hi, b_lo, b_hi,
            sm_scale, N, offs_n, offs_d, PREC,
        )  # fmt: skip
    for start_n in range(b_start, b_end, BLOCK_N):
        acc, m_i, l_i = _fwd_block(
            q, acc, m_i, l_i, start_n, kb, vb, skn, a_lo, a_hi, b_lo, b_hi,
            sm_scale, N, offs_n, offs_d, PREC,
        )  # fmt: skip
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    tl.store(
        O + h * sqh + offs_m[:, None] * sqn + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=rows[:, None],
    )
    tl.store(L + h * N + offs_m, m_i + tl.log(l_safe), mask=rows)


@triton.jit
def _bwd_pre(O, DO, Delta, sqh, sqn, N, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    off = h * sqh + offs_m[:, None] * sqn + offs_d[None, :]
    rows = offs_m[:, None] < N
    o = tl.load(O + off, mask=rows, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=rows, other=0.0).to(tl.float32)
    tl.store(Delta + h * N + offs_m, tl.sum(o * do, 1), mask=offs_m < N)


@triton.jit
def _dkdv_block(
    k,
    v,
    dk,
    dv,
    start_m,
    qb,
    dob,
    lb,
    deltab,
    sqn,
    QA_LO,
    QA_HI,
    QB_LO,
    QB_HI,
    offs_n,
    offs_m_base,
    offs_d,
    sm_scale,
    N,
    PREC: tl.constexpr,
):
    offs_m = start_m + offs_m_base
    rows = offs_m < N
    q = tl.load(
        qb + offs_m[:, None] * sqn + offs_d[None, :], mask=rows[:, None], other=0.0
    )
    do = tl.load(
        dob + offs_m[:, None] * sqn + offs_d[None, :], mask=rows[:, None], other=0.0
    )
    lse = tl.load(lb + offs_m, mask=rows, other=0.0)
    delta = tl.load(deltab + offs_m, mask=rows, other=0.0)
    a_lo = tl.load(QA_LO + offs_m, mask=rows, other=N)
    a_hi = tl.load(QA_HI + offs_m, mask=rows, other=-1)
    b_lo = tl.load(QB_LO + offs_m, mask=rows, other=N)
    b_hi = tl.load(QB_HI + offs_m, mask=rows, other=-1)
    allowed = _in_ranges(offs_n, a_lo, a_hi, b_lo, b_hi) & (offs_n[None, :] < N)
    s = tl.dot(q, tl.trans(k), input_precision=PREC) * sm_scale
    p = tl.where(allowed, tl.exp(s - lse[:, None]), 0.0)
    dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision=PREC)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    ds = (p * (dp - delta[:, None]) * sm_scale).to(q.dtype)
    dk += tl.dot(tl.trans(ds), q, input_precision=PREC)
    return dk, dv


@triton.jit
def _bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    L,
    Delta,
    QA_LO,
    QA_HI,
    QB_LO,
    QB_HI,
    KA_LO,
    KA_HI,
    KB_LO,
    KB_HI,
    sm_scale,
    sqh,
    sqn,
    skh,
    skn,
    N,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    start_n = tl.program_id(0)
    hk = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    cols = offs_n < N
    k = tl.load(
        K + hk * skh + offs_n[:, None] * skn + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    v = tl.load(
        V + hk * skh + offs_n[:, None] * skn + offs_d[None, :],
        mask=cols[:, None],
        other=0.0,
    )
    ka_lo = tl.load(KA_LO + offs_n, mask=cols, other=N)
    ka_hi = tl.load(KA_HI + offs_n, mask=cols, other=-1)
    kb_lo = tl.load(KB_LO + offs_n, mask=cols, other=N)
    kb_hi = tl.load(KB_HI + offs_n, mask=cols, other=-1)
    # Range B of a key (chain rows, or its own segment) lies before range A (later segments).
    a_start, a_end, b_start, b_end = _loop_bounds(kb_lo, kb_hi, ka_lo, ka_hi, BLOCK_M)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    for g in range(GROUPS):
        h = hk * GROUPS + g
        qb = Q + h * sqh
        dob = DO + h * sqh
        lb = L + h * N
        deltab = Delta + h * N
        for start_m in range(a_start, a_end, BLOCK_M):
            dk, dv = _dkdv_block(
                k, v, dk, dv, start_m, qb, dob, lb, deltab, sqn, QA_LO, QA_HI, QB_LO,
                QB_HI, offs_n, offs_m_base, offs_d, sm_scale, N, PREC,
            )  # fmt: skip
        for start_m in range(b_start, b_end, BLOCK_M):
            dk, dv = _dkdv_block(
                k, v, dk, dv, start_m, qb, dob, lb, deltab, sqn, QA_LO, QA_HI, QB_LO,
                QB_HI, offs_n, offs_m_base, offs_d, sm_scale, N, PREC,
            )  # fmt: skip
    tl.store(
        DK + hk * skh + offs_n[:, None] * skn + offs_d[None, :],
        dk.to(DK.dtype.element_ty),
        mask=cols[:, None],
    )
    tl.store(
        DV + hk * skh + offs_n[:, None] * skn + offs_d[None, :],
        dv.to(DV.dtype.element_ty),
        mask=cols[:, None],
    )


@triton.jit
def _dq_block(
    q,
    do,
    dq,
    lse,
    delta,
    start_n,
    kb,
    vb,
    skn,
    a_lo,
    a_hi,
    b_lo,
    b_hi,
    sm_scale,
    N,
    offs_n,
    offs_d,
    PREC: tl.constexpr,
):
    cols = start_n + offs_n
    k = tl.load(
        kb + cols[:, None] * skn + offs_d[None, :], mask=cols[:, None] < N, other=0.0
    )
    v = tl.load(
        vb + cols[:, None] * skn + offs_d[None, :], mask=cols[:, None] < N, other=0.0
    )
    allowed = _in_ranges(cols, a_lo, a_hi, b_lo, b_hi) & (cols[None, :] < N)
    s = tl.dot(q, tl.trans(k), input_precision=PREC) * sm_scale
    p = tl.where(allowed, tl.exp(s - lse[:, None]), 0.0)
    dp = tl.dot(do, tl.trans(v), input_precision=PREC)
    ds = (p * (dp - delta[:, None]) * sm_scale).to(k.dtype)
    dq += tl.dot(ds, k, input_precision=PREC)
    return dq


@triton.jit
def _bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    L,
    Delta,
    QA_LO,
    QA_HI,
    QB_LO,
    QB_HI,
    sm_scale,
    sqh,
    sqn,
    skh,
    skn,
    N,
    GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // GROUPS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    rows = offs_m < N
    off = h * sqh + offs_m[:, None] * sqn + offs_d[None, :]
    q = tl.load(Q + off, mask=rows[:, None], other=0.0)
    do = tl.load(DO + off, mask=rows[:, None], other=0.0)
    lse = tl.load(L + h * N + offs_m, mask=rows, other=0.0)
    delta = tl.load(Delta + h * N + offs_m, mask=rows, other=0.0)
    a_lo = tl.load(QA_LO + offs_m, mask=rows, other=N)
    a_hi = tl.load(QA_HI + offs_m, mask=rows, other=-1)
    b_lo = tl.load(QB_LO + offs_m, mask=rows, other=N)
    b_hi = tl.load(QB_HI + offs_m, mask=rows, other=-1)
    kb = K + hk * skh
    vb = V + hk * skh
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    a_start, a_end, b_start, b_end = _loop_bounds(a_lo, a_hi, b_lo, b_hi, BLOCK_N)
    for start_n in range(a_start, a_end, BLOCK_N):
        dq = _dq_block(
            q, do, dq, lse, delta, start_n, kb, vb, skn, a_lo, a_hi, b_lo, b_hi,
            sm_scale, N, offs_n, offs_d, PREC,
        )  # fmt: skip
    for start_n in range(b_start, b_end, BLOCK_N):
        dq = _dq_block(
            q, do, dq, lse, delta, start_n, kb, vb, skn, a_lo, a_hi, b_lo, b_hi,
            sm_scale, N, offs_n, offs_d, PREC,
        )  # fmt: skip
    tl.store(DQ + off, dq.to(DQ.dtype.element_ty), mask=rows[:, None])


def _rows_layout(t: torch.Tensor) -> torch.Tensor:
    """[H, N, D] head-major or token-major ([N, H, D] memory, as attention modules hold it)."""
    H, N, D = t.shape
    return t if t.stride() in ((N * D, D, 1), (D, H * D, 1)) else t.contiguous()


def _empty_as(t: torch.Tensor, dtype=None) -> torch.Tensor:
    return torch.empty_strided(
        t.shape, t.stride(), dtype=dtype or t.dtype, device=t.device
    )


class _RangedAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ranges, scale):
        # q [H, N, D], k/v [HK, N, D]. Kept in the caller's layout and the grads returned in
        # it: a head-major grad under a token-major view would reach the q/k norms
        # non-contiguous, which fused norm kernels may not handle.
        q, k = _rows_layout(q), _rows_layout(k)
        if v.stride() != k.stride():
            v = _empty_as(k).copy_(v)
        H, N, D = q.shape
        groups = H // k.shape[0]
        fwd_t, _, _ = _tiles(D)
        o = _empty_as(q)
        lse = torch.empty((H, N), device=q.device, dtype=torch.float32)
        _fwd[(triton.cdiv(N, fwd_t["BLOCK_M"]), H)](
            q, k, v, o, lse, *ranges[:4], scale, q.stride(0), q.stride(1),
            k.stride(0), k.stride(1), N, GROUPS=groups, HEAD_DIM=D, PREC=_prec(q), **fwd_t,
        )  # fmt: skip
        ctx.save_for_backward(q, k, v, o, lse, *ranges)
        ctx.scale = scale
        ctx.groups = groups
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, *ranges = ctx.saved_tensors
        H, N, D = q.shape
        if do.stride() != q.stride():
            do = _empty_as(q).copy_(do)
        _, kv_t, q_t = _tiles(D)
        delta = torch.empty((H, N), device=q.device, dtype=torch.float32)
        _bwd_pre[(triton.cdiv(N, 16), H)](
            o, do, delta, q.stride(0), q.stride(1), N, HEAD_DIM=D, BLOCK_M=16,
            num_warps=4,
        )  # fmt: skip
        dq = _empty_as(q)
        dk = _empty_as(k)
        dv = _empty_as(k)
        _bwd_dkdv[(triton.cdiv(N, kv_t["BLOCK_N"]), k.shape[0])](
            q, k, v, do, dk, dv, lse, delta, *ranges, ctx.scale, q.stride(0),
            q.stride(1), k.stride(0), k.stride(1), N, GROUPS=ctx.groups, HEAD_DIM=D, PREC=_prec(q),
            **kv_t,
        )  # fmt: skip
        _bwd_dq[(triton.cdiv(N, q_t["BLOCK_M"]), H)](
            q, k, v, do, dq, lse, delta, *ranges[:4], ctx.scale, q.stride(0),
            q.stride(1), k.stride(0), k.stride(1), N, GROUPS=ctx.groups, HEAD_DIM=D, PREC=_prec(q),
            **q_t,
        )  # fmt: skip
        return dq, dk, dv, None, None


def ranged_attention(q, k, v, ranges, scale):
    """q [H, N, D], k/v [HK, N, D]; ranges = (qa_lo, qa_hi, qb_lo, qb_hi, ka_lo, ka_hi,
    kb_lo, kb_hi), int32 [N] each. Returns [H, N, D]."""
    return _RangedAttn.apply(q, k, v, tuple(ranges), float(scale))


def _attention_reference(q, k, v, ranges, scale):
    """Dense fp32 reference of ranged_attention, for tests."""
    H, N, _ = q.shape
    qa_lo, qa_hi, qb_lo, qb_hi = (r.long() for r in ranges[:4])
    j = torch.arange(N, device=q.device)[None, :]
    allowed = ((j >= qa_lo[:, None]) & (j <= qa_hi[:, None])) | (
        (j >= qb_lo[:, None]) & (j <= qb_hi[:, None])
    )
    groups = H // k.shape[0]
    kf = k.float().repeat_interleave(groups, 0)
    vf = v.float().repeat_interleave(groups, 0)
    s = torch.einsum("hnd,hmd->hnm", q.float(), kf) * scale
    s = s.masked_fill(~allowed[None], float("-inf"))
    return torch.einsum("hnm,hmd->hnd", torch.softmax(s, -1), vf)
