# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.collection.calc_meta import AttnArg, FA4AttnArg

is_fa4_installed = False
try:
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd

    is_fa4_installed = True
except ImportError:
    pass

if is_fa4_installed:
    from .fa4_utils import load_precompiled_ffa_fa4

    load_precompiled_ffa_fa4()


# ---------------------------------------------------------------------------
# Triton kernel: per-K-block max score and optional block LSE
# Based on _flash_max_score_kernel from flash_idx.py — causal, large blocks,
# multi-stage pipelining.
# ---------------------------------------------------------------------------

@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_Q": 128, "BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["qk_head_dim", "block_size", "HAS_BLOCK_LSE", "HAS_Q_OFFSETS"],
)
@triton.jit
def _flash_block_score_kernel(
    q_ptr,
    k_ptr,
    sink_ptr,
    score_ptr,
    block_lse_ptr,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqblocks,
    q_offsets_ptr,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    block_size: tl.constexpr,
    sm_scale,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_sink_h,
    stride_sink_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_bl_h,
    stride_bl_q,
    stride_bl_k,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_KD: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_BLOCK_LSE: tl.constexpr,
    HAS_Q_OFFSETS: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    tl.static_assert(BLOCK_SIZE_K >= block_size)
    BLOCKS_PER_K_BLOCK: tl.constexpr = BLOCK_SIZE_K // block_size

    pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    if HAS_Q_OFFSETS:
        k_start = tl.load(cu_seqlens_k + pid_b)
        k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
        q_offset = tl.load(q_offsets_ptr + pid_b)
    else:
        k_start = q_start
        k_len = q_len
        q_offset = 0

    block_num = tl.load(cu_seqblocks + pid_b + 1) - tl.load(cu_seqblocks + pid_b)

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_k_n + pid_kh * stride_k_h,
        shape=(qk_head_dim, k_len),
        strides=(stride_k_d, stride_k_n),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KD, BLOCK_SIZE_K),
        order=(0, 1),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + q_start * stride_s_q + pid_h * stride_s_h,
        shape=(q_len, block_num),
        strides=(stride_s_q, stride_s_k),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
        order=(1, 0),
    )
    if HAS_BLOCK_LSE:
        bl_ptrs = tl.make_block_ptr(
            base=block_lse_ptr + q_start * stride_bl_q + pid_h * stride_bl_h,
            shape=(q_len, block_num),
            strides=(stride_bl_q, stride_bl_k),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK),
            order=(1, 0),
        )

    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + q_offset
    off_k = tl.arange(0, BLOCK_SIZE_K)

    diag_start = (pid_q * BLOCK_SIZE_Q + q_offset) // BLOCK_SIZE_K * BLOCK_SIZE_K
    hi = min(k_len, (pid_q + 1) * BLOCK_SIZE_Q + q_offset)

    for i in tl.range(0, hi, BLOCK_SIZE_K, num_stages=3):
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        qk = tl.dot(q, k) * sm_scale_log2e
        if i >= diag_start:
            qk = tl.where(off_q[:, None] >= (i + off_k)[None, :], qk, float("-inf"))

        qk_3d = tl.reshape(qk, (BLOCK_SIZE_Q, BLOCKS_PER_K_BLOCK, block_size), can_reorder=False)
        score = tl.max(qk_3d, axis=2)
        tl.store(s_ptrs, score.to(score_ptr.dtype.element_ty), boundary_check=(0, 1))

        if HAS_BLOCK_LSE:
            p = tl.exp2(qk_3d - score[:, :, None])
            if i + BLOCK_SIZE_K > k_len:
                off_k_r = tl.reshape(off_k, (BLOCKS_PER_K_BLOCK, block_size))
                k_valid = (i + off_k_r)[None, :, :] < k_len
                p = tl.where(k_valid, p, 0.0)
            s = tl.sum(p, axis=2)
            safe_m = tl.where(score > float("-inf"), score, 0.0)
            blse = safe_m * 0.6931471806 + tl.log(tl.where(s > 0.0, s, 1e-20))
            blse = tl.where(s > 0.0, blse, float("-inf"))
            tl.store(bl_ptrs, blse.to(block_lse_ptr.dtype.element_ty), boundary_check=(0, 1))
            bl_ptrs = tl.advance(bl_ptrs, (0, BLOCKS_PER_K_BLOCK))

        s_ptrs = tl.advance(s_ptrs, (0, BLOCKS_PER_K_BLOCK))
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))


def _triton_block_score(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    k_sparse_block_size: int = 128,
    return_block_lse: bool = False,
    sink: torch.Tensor | None = None,
    q_offsets: torch.Tensor | None = None,
    score_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-K-block max scores and optionally per-K-block LSE.

    Uses the same causal flash-scoring approach as ``_flash_max_score_kernel``
    (large blocks, multi-stage pipelining, causal loop bound).

    When ``q_offsets`` is provided, Q and K may have different cu_seqlens
    (e.g. CP>1 where Q is a local shard but K is the full sequence).
    ``q_offsets[d]`` gives the starting position of local Q within doc *d*'s
    full sequence, used for correct causal masking.

    Returns:
        (block_max, block_lse) in shape (total_q, num_heads_q, n_kblocks).
        block_max: per-block max(QK * sm_scale * log2e) in log2 scale.
        block_lse: ln(sum(exp(sm_scale * QK))) per block, or None.
    """
    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    gqa_group_size = num_heads_q // num_heads_kv
    batch_size = cu_seqlens_q.shape[0] - 1

    doc_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    doc_blocks = (doc_lens_k + k_sparse_block_size - 1) // k_sparse_block_size
    cu_seqblocks = torch.zeros(batch_size + 1, dtype=torch.int32, device=q.device)
    cu_seqblocks[1:] = torch.cumsum(doc_blocks, dim=0)
    max_seqblock = int(doc_blocks.max().item())

    _score_dtype = score_dtype if score_dtype is not None else q.dtype
    score = torch.full(
        (num_heads_q, total_q, max_seqblock),
        float("-inf"),
        dtype=_score_dtype,
        device=q.device,
    )
    block_lse = None
    if return_block_lse:
        block_lse = torch.full(
            (num_heads_q, total_q, max_seqblock),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

    has_q_offsets = q_offsets is not None
    if has_q_offsets:
        doc_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = int(doc_lens_q.max().item())
    else:
        max_seqlen_q = int(doc_lens_k.max().item())
        q_offsets = cu_seqlens_q  # dummy, not accessed

    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads_q)

    _flash_block_score_kernel[grid](
        q,
        k,
        sink,
        score,
        block_lse if block_lse is not None else score,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqblocks,
        q_offsets,
        num_heads_q,
        gqa_group_size,
        head_dim,
        k_sparse_block_size,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        (block_lse if block_lse is not None else score).stride(0),
        (block_lse if block_lse is not None else score).stride(1),
        (block_lse if block_lse is not None else score).stride(2),
        HAS_BLOCK_LSE=return_block_lse,
        HAS_Q_OFFSETS=has_q_offsets,
    )

    # Permute from (num_heads, total_q, n_kblocks) to (total_q, num_heads, n_kblocks)
    score = score.permute(1, 0, 2).contiguous()
    if block_lse is not None:
        block_lse = block_lse.permute(1, 0, 2).contiguous()

    return score, block_lse


def flash_attn_fwd_supports_max_score_out() -> bool:
    """True if installed ``flash_attn_cute._flash_attn_fwd`` accepts ``max_score_out``."""
    if not is_fa4_installed:
        return False
    return "max_score_out" in inspect.signature(_flash_attn_fwd).parameters


def flash_attn_fwd_supports_block_lse_out() -> bool:
    """True if installed ``flash_attn_cute._flash_attn_fwd`` accepts ``block_lse_out``."""
    if not is_fa4_installed:
        return False
    return "block_lse_out" in inspect.signature(_flash_attn_fwd).parameters


def fa4_max_score_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    k_sparse_block_size: int = 128,
    max_seqlen_k: int | None = None,
) -> tuple[int, int, int, int]:
    """Shape for ``max_score_out`` as passed to FA4 (with batch dim): ``(1, num_head, seqlen_q, n_k_chunks)``.

    When ``max_seqlen_k`` is provided (per-doc mode), the K-block dimension
    is ``ceil(max_seqlen_k / k_sparse_block_size)`` instead of using the full
    ``k.shape[0]``.
    """
    seqlen_q, num_head = q.shape[0], q.shape[1]
    eff_k = max_seqlen_k if max_seqlen_k is not None else k.shape[0]
    n_chunks = (eff_k + k_sparse_block_size - 1) // k_sparse_block_size
    return (1, num_head, seqlen_q, n_chunks)


@torch.no_grad()
def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink_layout: AttnSinkLayout = "sh",
    max_score_out: Optional[torch.Tensor] = None,
    return_max_score: bool = False,
    block_lse_out: Optional[torch.Tensor] = None,
    return_block_lse: bool = False,
    k_sparse_block_size: int = 128,
    max_seqlen_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert is_fa4_installed, "FlashAttn4 is not installed"
    assert isinstance(attn_arg, FA4AttnArg), "FA4 is only supported for FA4AttnArg"

    want_max_score = return_max_score or max_score_out is not None
    want_block_lse = return_block_lse or block_lse_out is not None

    _softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale

    # --- FA4 base forward (always without block outputs for speed) ---
    fa4_args = attn_arg.to_fa4_args(is_bwd=False)
    block_sparse = fa4_args["linear_k_block_sparse_mask"]

    q_b, k_b, v_b = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    fwd_kw: dict = dict(
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=sink,
        softcap=softcap,
        num_splits=1,
        pack_gqa=False,
        mask_mod=None,
        return_lse=True,
        block_sparse_tensors=block_sparse,
        aux_tensors=fa4_args["aux_tensors"],
    )

    # Pre-allocated block outputs passed by caller bypass the Triton path
    if max_score_out is not None:
        fwd_kw["max_score_out"] = max_score_out
        fwd_kw["k_sparse_block_size"] = k_sparse_block_size
    if block_lse_out is not None:
        fwd_kw["block_lse_out"] = block_lse_out
        fwd_kw["k_sparse_block_size"] = k_sparse_block_size

    out, lse = _flash_attn_fwd(q_b, k_b, v_b, **fwd_kw)

    out = out.squeeze(0)
    lse = lse.squeeze(0).mT

    # --- Block outputs from pre-allocated buffers (legacy path) ---
    max_score_sqh: Optional[torch.Tensor] = None
    if max_score_out is not None:
        max_score_sqh = max_score_out.squeeze(0).permute(1, 0, 2).contiguous()

    block_lse_sqh: Optional[torch.Tensor] = None
    if block_lse_out is not None:
        block_lse_sqh = block_lse_out.squeeze(0).permute(1, 0, 2).contiguous()

    # --- Triton block scoring (replaces slow FA4 block outputs) ---
    if (want_max_score or want_block_lse) and max_score_out is None and block_lse_out is None:
        k_ranges = attn_arg.k_ranges
        q_ranges = attn_arg.q_ranges
        if k_ranges.is_cu_seqlens(k.shape[0]) and q_ranges.is_cu_seqlens(q.shape[0]):
            cu_k = torch.tensor(
                k_ranges.to_cu_seqlens(k.shape[0]),
                dtype=torch.int32,
                device=q.device,
            )
            cu_q = torch.tensor(
                q_ranges.to_cu_seqlens(q.shape[0]),
                dtype=torch.int32,
                device=q.device,
            )
            triton_score, triton_blse = _triton_block_score(
                q, k, cu_q, cu_k,
                softmax_scale=_softmax_scale,
                k_sparse_block_size=k_sparse_block_size,
                return_block_lse=want_block_lse,
                sink=sink,
            )
            if want_max_score:
                max_score_sqh = triton_score
            if want_block_lse and triton_blse is not None:
                block_lse_sqh = triton_blse
        else:
            # Fallback: ranges not cu_seqlens-compatible, use FA4 native (slow)
            if want_max_score and not flash_attn_fwd_supports_max_score_out():
                raise RuntimeError(
                    "return_max_score requires FlashAttention cute build with max_score_out support."
                )
            if want_block_lse and not flash_attn_fwd_supports_block_lse_out():
                raise RuntimeError(
                    "return_block_lse requires FlashAttention cute build with block_lse_out support."
                )
            shape_ms = fa4_max_score_shape(q, k, k_sparse_block_size=k_sparse_block_size, max_seqlen_k=max_seqlen_k)
            if want_max_score:
                ms_buf = torch.full(shape_ms, float("-inf"), dtype=torch.float32, device=q.device)
                fwd_kw["max_score_out"] = ms_buf
                fwd_kw["k_sparse_block_size"] = k_sparse_block_size
            if want_block_lse:
                bl_buf = torch.full(shape_ms, float("-inf"), dtype=torch.float32, device=q.device)
                fwd_kw["block_lse_out"] = bl_buf
                fwd_kw["k_sparse_block_size"] = k_sparse_block_size
            _, _ = _flash_attn_fwd(q_b, k_b, v_b, **fwd_kw)
            if want_max_score:
                max_score_sqh = ms_buf.squeeze(0).permute(1, 0, 2).contiguous()
            if want_block_lse:
                block_lse_sqh = bl_buf.squeeze(0).permute(1, 0, 2).contiguous()

    return out, lse, max_score_sqh, block_lse_sqh


@torch.no_grad()
def fa4_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_arg: AttnArg,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    sink_layout: AttnSinkLayout = "sh",
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert is_fa4_installed, "FA4 backend is not installed"
    assert sink is None, "FA4 backend does not support leanable sink"
    assert isinstance(attn_arg, FA4AttnArg), "FA4 is only supported for FA4AttnArg"

    fa4_args = attn_arg.to_fa4_args(is_bwd=True)

    # Rearrange q,k,v,o,do: (s, h, d) -> (1, s, h, d)
    q, k, v, o, do = [x.unsqueeze(0) for x in (q, k, v, o, do)]

    # Rearrange lse: (s, h) -> (1, h, s)
    lse = lse.mT.unsqueeze(0).contiguous()

    dq, dk, dv = _flash_attn_bwd(
        q=q,
        k=k,
        v=v,
        out=o,
        dout=do,
        lse=lse,
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True,  # NOTE: to eanble arbitrary mask functionality
        softcap=softcap,
        block_sparse_tensors=fa4_args["linear_q_block_sparse_mask"],
        aux_tensors=fa4_args["aux_tensors"],
        deterministic=deterministic,
    )
    dsink = None

    # Rearrange dq,dk,dv: (1, s, h, d) -> (s, h, d)
    dq, dk, dv = dq.squeeze(0), dk.squeeze(0), dv.squeeze(0)

    return dq, dk, dv, dsink


class FA4AttnFunc(torch.autograd.Function):
    """Autograd function for FA4 backend with arbitrary mask support.

    Uses FA4AttnArg from calc_meta.py to build FA4 args.
    """

    # Cache for reusing FA4AttnArg across calls
    _cached_fa4_attn_arg = None

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor | None,
        softmax_scale: float | None,
        softcap: float,
        reuse_attn_arg: bool = False,
        return_max_score: bool = False,
    ):
        softmax_scale = (
            q.shape[-1] ** (-0.5) if softmax_scale is None else softmax_scale
        )

        # Reuse cached FA4AttnArg if available and requested
        if reuse_attn_arg and FA4AttnFunc._cached_fa4_attn_arg is not None:
            fa4_attn_arg = FA4AttnFunc._cached_fa4_attn_arg  # type: ignore[unreachable]
        else:
            seqlen_q = q.shape[0]
            seqlen_k = k.shape[0]

            # Build AttnRanges from tensor (required by FA4AttnArg interface)
            q_ranges_list = q_ranges.cpu().tolist()
            k_ranges_list = k_ranges.cpu().tolist()

            # Build attn_type_map list
            if attn_type_map is None:
                attn_type_map_list = [0] * len(q_ranges_list)
            else:
                attn_type_map_list = attn_type_map.cpu().tolist()

            # Create FA4AttnArg (reuses _transfer_ffa_args_to_fa4_args from calc_meta.py)
            fa4_attn_arg = FA4AttnArg(
                q_ranges=AttnRanges.from_ranges(q_ranges_list),
                k_ranges=AttnRanges.from_ranges(k_ranges_list),
                attn_type_map=attn_type_map_list,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
            )
            # Cache for future reuse
            FA4AttnFunc._cached_fa4_attn_arg = fa4_attn_arg

        out, lse, max_sc, _block_lse = fa4_fwd(
            q=q,
            k=k,
            v=v,
            sink=None,
            attn_arg=fa4_attn_arg,
            softmax_scale=softmax_scale,
            softcap=softcap,
            return_max_score=return_max_score,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, q_ranges, k_ranges, attn_type_map)
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.fa4_attn_arg = fa4_attn_arg
        ctx.return_max_score = return_max_score

        if return_max_score:
            return out, lse, max_sc
        return out, lse

    @staticmethod
    def backward(ctx, *grad_outputs):
        q, k, v, out, lse, q_ranges, k_ranges, attn_type_map = ctx.saved_tensors
        if ctx.return_max_score:
            dout, dlse, _ = grad_outputs
        else:
            dout, dlse = grad_outputs

        # Call fa4_bwd
        dq, dk, dv, _ = fa4_bwd(
            do=dout,
            q=q,
            k=k,
            v=v,
            sink=None,
            o=out,
            lse=lse,
            attn_arg=ctx.fa4_attn_arg,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
        )

        # Return gradients for each input (None for non-tensor args)
        return dq, dk, dv, None, None, None, None, None, None, None


def ffa_fa4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
    *,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    reuse_attn_arg: bool = False,
    return_max_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FA4 backend version of flex_flash_attn_func for benchmarking.

    Similar interface to flex_flash_attn_func but uses FA4 backend with arbitrary mask support.
    Supports both forward and backward passes.

    Args:
        q (torch.Tensor): Query tensor with shape (seqlen_q, num_heads_q, head_dim).
        k (torch.Tensor): Key tensor with shape (seqlen_k, num_heads_k, head_dim).
        v (torch.Tensor): Value tensor with shape (seqlen_k, num_heads_k, head_dim).
        q_ranges (torch.Tensor): Query ranges tensor with shape (num_ranges, 2).
        k_ranges (torch.Tensor): Key ranges tensor with shape (num_ranges, 2).
        attn_type_map (torch.Tensor, optional): Attention type map tensor.
        softmax_scale (float, optional): Softmax scale.
        softcap (float): Softcap value.
        reuse_attn_arg (bool): If True, reuse the cached FA4AttnArg from previous call.
            Set to False for warmup/first call, then True for subsequent calls
            to measure only kernel time without FA4AttnArg creation overhead.
        return_max_score (bool): If True, returns per-K-block max scores from the FA4
            kernel (SM100).  Values are raw ``max(QK)`` without ``sm_scale``.

    Returns:
        (out, lse) or (out, lse, max_score) with ``max_score`` shape
        ``(seqlen_q, num_heads, ceil(seqlen_k / 128))`` when ``return_max_score`` is True.
    """
    return FA4AttnFunc.apply(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        softmax_scale,
        softcap,
        reuse_attn_arg,
        return_max_score,
    )
