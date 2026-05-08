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

from dataclasses import dataclass

import torch


@dataclass
class CalcAttnCustomAttribute:
    """Optional flags for :func:`~magi_attention.api.calc_attn` (FA4 / SM100 path).

    When ``return_block_max`` or ``return_block_lse`` is True, requires FA4 backend;
    see ``AttnForwardMeta.block_max`` / ``block_lse`` shapes.
    """

    return_block_max: bool = False
    return_block_max_dtype: torch.dtype = torch.float32
    return_block_lse: bool = False
    k_sparse_block_size: int = 128
    max_per_doc_seqlen_k: int | None = None


@dataclass
class AttnForwardMeta:
    """Attention forward metadata.

    Attributes:
        lse: Log-sum-exp of the attention weights. In a distributed setting, this is a
            local tensor where each device holds the LSE computed from its local query
            shards.
        max_logits: Maximum logits per query head. In a distributed setting,
            this is a replicated tensor where each device holds the global maximum
            computed across the entire sequence, ensuring consistency across all devices.
        block_max: Per-K-block max scores (FA4 ``max_score``), shape
            ``(seqlen_q, num_heads_q, ceil(seqlen_k / k_sparse_block_size))``,
            dtype controlled by ``CalcAttnCustomAttribute.return_block_max_dtype``
            (default float32); ``None`` if not requested.  When
            ``overlap_degree > 0``, the last dimension spans all stages
            concatenated in global KV order.
        block_lse: Per-K-block scaled LSE (FA4 ``block_lse_out``), same shape as
            ``block_max``; ``None`` if not requested.
    """

    lse: torch.Tensor | None
    max_logits: torch.Tensor | None
    block_max: torch.Tensor | None = None
    block_lse: torch.Tensor | None = None
