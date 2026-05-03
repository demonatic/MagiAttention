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

from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.common import CalcAttnCustomAttribute
from magi_attention.api.functools import (
    apply_padding,
    compute_pad_size,
    infer_varlen_mask_from_batch,
    pad_at_dim,
)
from magi_attention.api.magi_attn_interface import (
    calc_attn,
    dispatch,
    dist_attn_runtime_dict_mgr,
    get_position_ids,
    magi_attn_flex_key,
    magi_attn_varlen_key,
    make_flex_key_for_new_mask_after_dispatch,
    make_varlen_key_for_new_mask_after_dispatch,
    undispatch,
)
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    SequentialDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeKey,
    DistAttnRuntimeMgr,
    init_dist_attn_runtime_mgr,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import (
    INTERFACE,
    NAME,
    SKIP_WORLD_SIZE,
    DistTestBase,
    with_comms,
)
from magi_attention.testing.precision import (
    H100_MATMUL_MFU,
    H100_NVLINK_A2A_BWU,
    H100_NVLINK_BANDWIDTH,
    H100_TFLOPS_16,
)
from magi_attention.testing.utils import switch_deterministic_mode_decorator
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_calc_cost_factor,
    get_comm_cost_factor,
    is_list_value_all,
)


class TestInterfaceBaseWithWorldSize1(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend=self.backend)
            for _ in range(1)
        ]

        # -----    set up for hier comm   ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
            world_size_inter_node, world_size_intra_node = {
                1: (1, 1),
                2: (1, 2),
                3: (3, 1),
                4: (2, 2),
                5: (1, 5),
                6: (3, 2),
                7: (1, 7),
                8: (2, 4),
            }[self.world_size]
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            self.device_mesh = None

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    @property
    def seed(self) -> int:
        return 42 + self.world_size

    @with_comms
    @parameterize(
        "attn_config",
        [
            # full attn with seqlen 2k
            {
                NAME: "full_attn_2k_bs2",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                INTERFACE: "magi_attn",
                "batch_size": 2,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 2048],
                    ]
                ),
                "attn_type_mapping": 0,
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
                "chunk_size": 1024,
            },
            # full attn with seqlen 6k and batch size 3
            {
                NAME: "full_attn_6k_bs3",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                INTERFACE: "magi_attn",
                "batch_size": 3,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                    ]
                ),
                "attn_type_mapping": [0, 0, 0],
                "total_seqlen_q": 6144,
                "total_seqlen_k": 6144,
                "chunk_size": 1536,
            },
            # varlen full attn with total seqlen 1050
            {
                NAME: "flex_varlen_full_attn_1050",
                SKIP_WORLD_SIZE: [4, 8],
                INTERFACE: "magi_attn_flex",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "attn_type_mapping": 0,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
                "chunk_size": 257,
                "use_str_masktype": False,
            },
            {
                NAME: "varlen_full_attn_1050",
                SKIP_WORLD_SIZE: [4, 8],
                INTERFACE: "magi_attn_varlen",
                "test_make_new_key": True,
                "cu_seqlens_q": torch.tensor(
                    [0, 128, 256, 384, 512, 640, 768, 1050], dtype=torch.int32
                ),
                "cu_seqlens_k": torch.tensor(
                    [0, 128, 256, 384, 512, 640, 768, 1050], dtype=torch.int32
                ),
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "attn_type_mapping": [0] * 7,
                "new_cu_seqlens_q": torch.tensor([0, 512, 1050], dtype=torch.int32),
                "new_cu_seqlens_k": torch.tensor([0, 512, 1050], dtype=torch.int32),
                "new_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 512],
                        [512, 1050],
                    ]
                ),
                "new_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 512],
                        [512, 1050],
                    ]
                ),
                "new_attn_type_mapping": 0,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
                "chunk_size": 568,
            },
            # varlen block causal with total seqlen 960
            {
                NAME: "varlen_block_causal_960",
                SKIP_WORLD_SIZE: [7, 8],
                INTERFACE: "magi_attn_flex",
                "test_make_new_key": True,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 960],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [0, 256],
                        [0, 384],
                        [0, 512],
                        [512, 640],
                        [512, 768],
                        [768, 960],
                    ]
                ),
                "attn_type_mapping": [0, 1, 2, 3, 1, 2, 3],
                "new_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 512],
                        [512, 960],
                    ]
                ),
                "new_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 512],
                        [512, 960],
                    ]
                ),
                "new_attn_type_mapping": 1,
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 568,
                "use_str_masktype": True,
            },
            # cp_mesh and cp_group are both set, raise ValueError
            {
                NAME: "cp_mesh and cp_group testcase",
                SKIP_WORLD_SIZE: [1, 2, 3, 5, 7],
                INTERFACE: "set_mesh_and_group",
                "q_ranges": AttnRanges.from_ranges([[0, 960]]),
                "k_ranges": AttnRanges.from_ranges([[0, 960]]),
                "attn_type_mapping": [0],
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 568,
            },
            # test for invalid masktype
            # NOTE: it is a common typo that people write "casual" instead of "causal"
            {
                NAME: "cp_mesh and cp_group testcase",
                SKIP_WORLD_SIZE: [3, 5, 7],
                INTERFACE: "test_for_invalid_mask",
                "q_ranges": AttnRanges.from_ranges([[0, 960]]),
                "k_ranges": AttnRanges.from_ranges([[0, 960]]),
                "attn_type_mapping": [0],
                "attn_mask_type": ["casual"],
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 324,
            },
        ],
    )
    @parameterize(
        # TODO:
        #   1. test non-trivial algorithms
        #   2. profile real comm/calc factors
        "overlap_config",
        [
            # disable multi-stage overlap
            {
                NAME: "disable_mso",
                "enable": False,
            },
            # static, overlap degree = 4, min chunk size = 23
            {
                NAME: "static_od4_cz23",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 13,
                "max_num_chunks": 52,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
            # dynamic, min chunk size = 56, no max overlap degree limit
            {
                NAME: "dynamic_cz56",
                "enable": True,
                "mode": AttnOverlapMode.DYNAMIC,
                "degree": None,
                "dynamic_max_degree": None,
                "min_chunk_size": 12,
                "max_num_chunks": 65,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
        ],
    )
    @parameterize(
        "num_heads",
        [(6, 2)],  # gqa
    )
    @parameterize(
        "head_dim",
        [128],
    )
    @parameterize(
        "dtype",
        [torch.bfloat16],
    )
    def test_interface(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: tuple[int, int],  # (nhq, nhkv)
        head_dim: int,
        dtype: torch.dtype,
    ):
        # -----    skip for world size   ---- #

        if (
            attn_config.get(SKIP_WORLD_SIZE, [])
            and self.world_size in attn_config[SKIP_WORLD_SIZE]
        ):
            return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=} | \n\n{overlap_config=}"

        test_case = (
            f"world_size=[{self.world_size}] x "
            f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x "
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})]"
        )

        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        interface: str = attn_config["interface"]
        attn_type_mapping: int | list[int] = attn_config["attn_type_mapping"]
        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]
        num_heads_q, num_heads_kv = num_heads

        dist_attn_config = DistAttnConfig(
            # TODO: test other dispatch algs
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                **{k: v for k, v in overlap_config.items() if k not in (NAME,)},
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=num_heads_q,
                    head_dim=head_dim,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(self.world_size),
                ),
            ),
        )

        # ----- init input data and module ----- #

        x = torch.randn(
            total_seqlen_q,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=True,
        )

        # --------- calculate pad size --------- #

        pad_size = compute_pad_size(total_seqlen_q, self.world_size, chunk_size)

        # ------ calculate attn_mask_type ------ #

        if isinstance(attn_type_mapping, list):
            attn_mask_type = list(map(AttnMaskType.from_int_type, attn_type_mapping))
        else:
            attn_mask_type = [AttnMaskType.from_int_type(attn_type_mapping)] * len(
                q_ranges
            )

        # ------ test interface ------ #

        match interface:
            case "magi_attn":  # [b, s, nh, hd]
                assert is_list_value_all(
                    attn_mask_type, AttnMaskType.FULL
                ) or is_list_value_all(
                    attn_mask_type, AttnMaskType.CAUSAL
                ), "we need to check varlen interface, which supports full or causal now"
                is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

                batch_size = attn_config["batch_size"]
                cu_seqlens_q, cu_seqlens_k = infer_varlen_mask_from_batch(
                    batch_size, attn_config["total_seqlen_q"] // batch_size
                )

                dist_attn_runtime_key = magi_attn_varlen_key(
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    causal=is_causal,
                    dist_attn_config=dist_attn_config,
                )
            case "magi_attn_varlen":
                assert is_list_value_all(
                    attn_mask_type, AttnMaskType.FULL
                ) or is_list_value_all(
                    attn_mask_type, AttnMaskType.CAUSAL
                ), "we need to check varlen interface, which supports full or causal now"
                is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

                cu_seqlens_q = attn_config["cu_seqlens_q"]
                cu_seqlens_k = attn_config["cu_seqlens_k"]
                dist_attn_runtime_key = magi_attn_varlen_key(
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    causal=is_causal,
                    dist_attn_config=dist_attn_config,
                )
            case "magi_attn_flex":
                use_str_masktype: bool = attn_config["use_str_masktype"]
                dist_attn_runtime_key = magi_attn_flex_key(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_mask_type=[masktype.value for masktype in attn_mask_type]
                    if use_str_masktype
                    else attn_mask_type,
                    total_seqlen_q=total_seqlen_q,
                    total_seqlen_k=total_seqlen_k,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    dist_attn_config=dist_attn_config,
                )
                local_x_padded = dispatch(x, key=dist_attn_runtime_key)
            case "set_mesh_and_group":
                if magi_attention.comm.is_hierarchical_comm_enable():
                    with pytest.raises(AssertionError):
                        dist_attn_runtime_key = magi_attn_flex_key(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_mask_type=attn_mask_type,
                            total_seqlen_q=total_seqlen_q,
                            total_seqlen_k=total_seqlen_k,
                            num_heads_q=num_heads_q,
                            num_heads_kv=num_heads_kv,
                            head_dim=head_dim,
                            pad_size=pad_size,
                            chunk_size=chunk_size,
                            cp_group_or_mesh=self.nccl_group,
                            dist_attn_config=dist_attn_config,
                        )
                else:
                    with pytest.raises(ValueError):
                        dist_attn_runtime_key = magi_attn_flex_key(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_mask_type=attn_mask_type,
                            total_seqlen_q=total_seqlen_q,
                            total_seqlen_k=total_seqlen_k,
                            num_heads_q=num_heads_q,
                            num_heads_kv=num_heads_kv,
                            head_dim=head_dim,
                            pad_size=pad_size,
                            chunk_size=chunk_size,
                            cp_group_or_mesh=self.device_mesh,
                            dist_attn_config=dist_attn_config,
                        )
                return
            case "test_for_invalid_mask":
                invalid_mask_type = attn_config["attn_mask_type"]
                with pytest.raises(ValueError):
                    dist_attn_runtime_key = magi_attn_flex_key(
                        q_ranges=q_ranges,
                        k_ranges=k_ranges,
                        attn_mask_type=invalid_mask_type,
                        total_seqlen_q=total_seqlen_q,
                        total_seqlen_k=total_seqlen_k,
                        num_heads_q=num_heads_q,
                        num_heads_kv=num_heads_kv,
                        head_dim=head_dim,
                        pad_size=pad_size,
                        chunk_size=chunk_size,
                        cp_group_or_mesh=self.device_mesh
                        if magi_attention.comm.is_hierarchical_comm_enable()
                        else self.nccl_group,
                        dist_attn_config=dist_attn_config,
                    )
                return
            case _:
                raise ValueError(f"Invalid interface: {interface}")

        # -----    compute dist attn runtime mgr   ---- #

        dist_attn_runtime_mgr: DistAttnRuntimeMgr = dist_attn_runtime_dict_mgr[
            dist_attn_runtime_key
        ]

        # -------   calc ref_attn_runtime_mgr -------- #

        if pad_size > 0:
            q_ranges, k_ranges, attn_mask_type = apply_padding(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen=total_seqlen_q,
                pad_size=pad_size,
            )

        ref_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q + pad_size,
            total_seqlen_k=total_seqlen_k + pad_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            cp_mesh=self.device_mesh,
            dist_attn_config=dist_attn_config,
        )

        # -------   check mgr equality to ref -------- #

        assert (
            dist_attn_runtime_mgr == ref_attn_runtime_mgr
        ), f"For {test_case=}, the {dist_attn_runtime_mgr=} is not equal to the {ref_attn_runtime_mgr=}."

        # -------   test position ids -------- #

        if interface == "magi_attn_flex":
            global_x_padded = pad_at_dim(x, 0, pad_size)

            #  -----  get position_ids and check  -----  #

            position_ids = get_position_ids(dist_attn_runtime_key)
            position_ids = position_ids[
                position_ids < total_seqlen_q - 1
            ]  # remove padded id
            valid_length = position_ids.size(0)

            self.assertTrue(
                torch.equal(
                    local_x_padded[:valid_length], global_x_padded[position_ids]
                )
            )

        # -------   test make new key -------- #

        if attn_config.get("test_make_new_key", False):
            new_q_ranges: AttnRanges = attn_config["new_q_ranges"]
            new_k_ranges: AttnRanges = attn_config["new_k_ranges"]
            new_attn_type_mapping: int | list[int] = attn_config[
                "new_attn_type_mapping"
            ]
            if isinstance(new_attn_type_mapping, list):
                new_attn_mask_type = list(
                    map(AttnMaskType.from_int_type, new_attn_type_mapping)
                )
            else:
                new_attn_mask_type = [
                    AttnMaskType.from_int_type(new_attn_type_mapping)
                ] * len(new_q_ranges)

            new_dist_attn_config = DistAttnConfig()

            match interface:
                case "magi_attn_varlen":
                    new_cu_seqlens_q = attn_config["new_cu_seqlens_q"]
                    new_cu_seqlens_k = attn_config["new_cu_seqlens_k"]

                    new_key: DistAttnRuntimeKey = (
                        make_varlen_key_for_new_mask_after_dispatch(
                            cu_seqlens_q=new_cu_seqlens_q,
                            cu_seqlens_k=new_cu_seqlens_k,
                            key_for_dispatch=dist_attn_runtime_key,
                            dist_attn_config=new_dist_attn_config,
                        )
                    )
                case "magi_attn_flex":
                    new_key: DistAttnRuntimeKey = (  # type: ignore[no-redef]
                        make_flex_key_for_new_mask_after_dispatch(
                            q_ranges=new_q_ranges,
                            k_ranges=new_k_ranges,
                            attn_mask_type=new_attn_mask_type,
                            key_for_dispatch=dist_attn_runtime_key,
                            dist_attn_config=new_dist_attn_config,
                        )
                    )
                case _:
                    raise ValueError(
                        f"Invalid interface for make_new_key test: {interface}"
                    )

            if pad_size > 0:
                new_q_ranges, new_k_ranges, new_attn_mask_type = apply_padding(
                    q_ranges=new_q_ranges,
                    k_ranges=new_k_ranges,
                    attn_mask_type=new_attn_mask_type,
                    total_seqlen=total_seqlen_q,
                    pad_size=pad_size,
                )

            # check new key
            assert new_key.q_ranges == new_q_ranges
            assert new_key.k_ranges == new_k_ranges
            assert new_key.attn_mask_type == tuple(new_attn_mask_type)
            assert new_key.total_seqlen_q == total_seqlen_q + pad_size
            assert new_key.total_seqlen_k == total_seqlen_k + pad_size
            assert new_key.pad_size == pad_size
            assert new_key.chunk_size == chunk_size
            assert new_key.dist_attn_config == new_dist_attn_config

            new_mgr: DistAttnRuntimeMgr = dist_attn_runtime_dict_mgr[new_key]
            ref_new_mgr = init_dist_attn_runtime_mgr(
                q_ranges=new_q_ranges,
                k_ranges=new_k_ranges,
                attn_mask_type=new_attn_mask_type,
                total_seqlen_q=total_seqlen_q + pad_size,
                total_seqlen_k=total_seqlen_k + pad_size,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                chunk_size=chunk_size,
                cp_group=self.nccl_group,
                cp_mesh=self.device_mesh,
                dist_attn_config=new_dist_attn_config,
                ref_dispatch_meta_q=dist_attn_runtime_mgr.dispatch_meta_q,
                ref_dispatch_meta_k=dist_attn_runtime_mgr.dispatch_meta_k,
            )
            assert (
                new_mgr == ref_new_mgr
            ), f"For {test_case=}, the {new_mgr=} is not equal to the {ref_new_mgr=}."

    def _check_block_score(
        self, total_seqlen, num_docs, num_heads_q, num_heads_kv,
        head_dim=128, block_size_k=128, chunk_size=256, seed=42,
    ):
        """Run one block_max / block_lse correctness check against PyTorch ref."""
        import math, gc

        dtype = torch.bfloat16
        sm_scale = head_dim ** -0.5
        sm_scale_log2e = sm_scale * math.log2(math.e)

        torch.manual_seed(seed)
        raw = torch.randint(
            max(1, total_seqlen // (num_docs * 2)),
            total_seqlen // num_docs + 1,
            (num_docs,),
        )
        doc_lens = (raw.float() / raw.sum() * total_seqlen).int()
        doc_lens[-1] = total_seqlen - doc_lens[:-1].sum()
        cu_seqlens = torch.zeros(num_docs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = torch.cumsum(doc_lens.to(self.device), dim=0)
        max_seqlen = int(doc_lens.max().item())

        pad_size = compute_pad_size(total_seqlen, self.world_size, chunk_size)

        dist_attn_config = None
        if self.world_size > 1:
            cost_kwargs = dict(
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=num_heads_q,
                    head_dim=head_dim,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(self.world_size),
                ),
            )
            dist_attn_config = DistAttnConfig(
                dispatch_config=DispatchConfig(alg=SequentialDispatchAlg()),
                overlap_config=OverlapConfig(
                    enable=True,
                    mode=AttnOverlapMode.STATIC,
                    degree=4,
                    min_chunk_size=block_size_k,
                    max_num_chunks=64,
                    alg=UniformOverlapAlg(random_costs=False),
                    **cost_kwargs,
                ),
            )

        key_kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=self.device_mesh
            if magi_attention.comm.is_hierarchical_comm_enable()
            else self.nccl_group,
            causal=True,
        )
        if dist_attn_config is not None:
            key_kwargs["dist_attn_config"] = dist_attn_config
        key = magi_attn_varlen_key(**key_kwargs)

        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 1)
        full_q = torch.randn(
            total_seqlen, num_heads_q, head_dim,
            device=self.device, dtype=dtype,
        )
        full_k = torch.randn(
            total_seqlen, num_heads_kv, head_dim,
            device=self.device, dtype=dtype,
        )
        full_v = torch.randn(
            total_seqlen, num_heads_kv, head_dim,
            device=self.device, dtype=dtype,
        )

        local_q = dispatch(full_q, key=key)
        local_k = dispatch(full_k, key=key)
        local_v = dispatch(full_v, key=key)

        custom = CalcAttnCustomAttribute(
            return_block_max=True,
            return_block_lse=True,
            k_sparse_block_size=block_size_k,
        )
        _, meta = calc_attn(local_q, local_k, local_v, key, custom_attribute=custom)
        assert meta.block_max is not None
        assert meta.block_lse is not None

        global_block_max = undispatch(meta.block_max, key=key)
        global_block_lse = undispatch(meta.block_lse, key=key)

        # --- PyTorch reference (memory-efficient: block-by-block) ---
        n_kblocks = math.ceil(max_seqlen / block_size_k)
        ref_block_max = torch.full(
            (total_seqlen, num_heads_q, n_kblocks),
            float("-inf"), dtype=torch.float32, device=self.device,
        )
        ref_block_lse = torch.full(
            (total_seqlen, num_heads_q, n_kblocks),
            float("-inf"), dtype=torch.float32, device=self.device,
        )

        gqa_ratio = num_heads_q // num_heads_kv
        for d in range(num_docs):
            qs = cu_seqlens[d].item()
            qe = cu_seqlens[d + 1].item()
            doc_len = qe - qs
            q_doc = full_q[qs:qe].float()
            k_doc = full_k[qs:qe].float()
            if gqa_ratio > 1:
                k_doc = k_doc.repeat_interleave(gqa_ratio, dim=1)
            q_idx = torch.arange(doc_len, device=self.device)

            n_doc_blocks = math.ceil(doc_len / block_size_k)
            for b in range(n_doc_blocks):
                k_s = b * block_size_k
                k_e = min((b + 1) * block_size_k, doc_len)
                qk_block = torch.einsum(
                    'qhd,khd->hqk', q_doc, k_doc[k_s:k_e]
                )
                causal_mask = q_idx[:, None] < torch.arange(
                    k_s, k_e, device=self.device
                )[None, :]
                qk_block[:, causal_mask] = float("-inf")

                ref_block_max[qs:qe, :, b] = (
                    qk_block.max(dim=2).values.T * sm_scale_log2e
                )
                ref_block_lse[qs:qe, :, b] = torch.logsumexp(
                    qk_block * sm_scale, dim=2
                ).T

        # --- Compare ---
        n = min(n_kblocks, global_block_max.shape[2])
        label = f"{total_seqlen // 1024}K-{num_docs}doc-{num_heads_q}h"
        doc_lens_list = [int(cu_seqlens[i+1] - cu_seqlens[i]) for i in range(num_docs)]
        print(f"[_check_block_score] {label} | ws={self.world_size} "
              f"| doc_lens={doc_lens_list} | n_kblocks={n_kblocks}")

        for name, magi_out, ref_out in [
            ("block_max", global_block_max, ref_block_max),
            ("block_lse", global_block_lse, ref_block_lse),
        ]:
            magi_t = magi_out[:total_seqlen, :, :n]
            ref_t = ref_out[:, :, :n]

            ref_inf = ref_t == float("-inf")
            magi_inf = magi_t == float("-inf")
            disagree = (ref_inf != magi_inf).sum().item()
            assert disagree == 0, (
                f"[{label}] {name}: {disagree} -inf positions disagree"
            )

            valid = torch.isfinite(ref_t) & torch.isfinite(magi_t)
            assert valid.any(), f"[{label}] {name}: no valid entries"
            ref_v = ref_t[valid].double()
            magi_v = magi_t[valid].double()

            cos_sim = torch.nn.functional.cosine_similarity(
                ref_v.unsqueeze(0), magi_v.unsqueeze(0)
            ).item()
            print(f"  {name}: cos_sim={cos_sim:.10f}")
            assert cos_sim > 0.99999, (
                f"[{label}] {name} cosine similarity {cos_sim:.7f} < 0.99999"
            )
            torch.testing.assert_close(magi_v, ref_v, atol=1e-4, rtol=1e-4)

        del meta, ref_block_max, ref_block_lse
        gc.collect()
        torch.cuda.empty_cache()

    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_calc_attn_block_score_correctness(self):
        """Verify block_max and block_lse against PyTorch reference (multi-doc varlen causal)."""
        if not magi_attention.is_fa4_backend_enable():
            self.skipTest("FA4 backend not enabled")
        from magi_attention.functional import fa4 as fa4_mod

        if not fa4_mod.is_fa4_installed:
            self.skipTest("flash_attn not installed")

        configs = [
            # (total_seqlen, num_docs, num_heads_q, num_heads_kv)
            (8192,    4,  4, 4),
            (32768,   8,  4, 4),
            (192000,  8,  4, 4),
            (192000, 16,  1, 1),
        ]
        for i, (seqlen, ndocs, nhq, nhkv) in enumerate(configs):
            self._check_block_score(
                total_seqlen=seqlen, num_docs=ndocs,
                num_heads_q=nhq, num_heads_kv=nhkv, seed=42 + i,
            )

    @skip_if_lt_x_gpu(1)
    @with_comms
    def test_calc_attn_custom_attribute_non_fa4_raises(self):
        if magi_attention.is_fa4_backend_enable():
            self.skipTest("non-FA4 backend only")

        total_seqlen = 128
        num_heads_q, num_heads_kv = 4, 4
        head_dim = 64
        dtype = torch.bfloat16
        chunk_size = 128
        q_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
        pad_size = compute_pad_size(total_seqlen, self.world_size, chunk_size)
        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                alg=UniformOverlapAlg(),
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=num_heads_q,
                    head_dim=head_dim,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(self.world_size),
                ),
            ),
        )
        key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL],
            total_seqlen_q=total_seqlen,
            total_seqlen_k=total_seqlen,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=self.device_mesh
            if magi_attention.comm.is_hierarchical_comm_enable()
            else self.nccl_group,
            dist_attn_config=dist_attn_config,
        )
        x = torch.randn(total_seqlen, head_dim, device=self.device, dtype=dtype)
        local_x = dispatch(x, key=key)
        sq = local_x.shape[0]
        local_q = torch.randn(sq, num_heads_q, head_dim, device=self.device, dtype=dtype)
        local_k = torch.randn(sq, num_heads_kv, head_dim, device=self.device, dtype=dtype)
        local_v = torch.randn(sq, num_heads_kv, head_dim, device=self.device, dtype=dtype)
        with pytest.raises(ValueError, match="FA4"):
            calc_attn(
                local_q,
                local_k,
                local_v,
                key,
                custom_attribute=CalcAttnCustomAttribute(return_block_lse=True),
            )


class TestInterfaceWithWorldSize2(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_calc_attn_custom_attribute_overlap_block_shapes(self):
        """block_max / block_lse must be returned with correct shapes when
        overlap_degree > 0 (the restriction was lifted)."""
        if not magi_attention.is_fa4_backend_enable():
            self.skipTest("FA4 backend not enabled")
        from magi_attention.functional import fa4 as fa4_mod

        if not fa4_mod.is_fa4_installed:
            self.skipTest("flash_attn not installed")
        if not fa4_mod.flash_attn_fwd_supports_max_score_out():
            self.skipTest("FA build lacks max_score_out")
        if not fa4_mod.flash_attn_fwd_supports_block_lse_out():
            self.skipTest("FA build lacks block_lse_out")

        total_seqlen = 512
        num_heads_q, num_heads_kv = 4, 4
        head_dim = 64
        dtype = torch.bfloat16
        chunk_size = 256
        k_sparse_block_size = 128
        q_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
        k_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
        pad_size = compute_pad_size(total_seqlen, self.world_size, chunk_size)
        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                alg=UniformOverlapAlg(),
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=num_heads_q,
                    head_dim=head_dim,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(self.world_size),
                ),
            ),
        )
        key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL],
            total_seqlen_q=total_seqlen,
            total_seqlen_k=total_seqlen,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=self.device_mesh
            if magi_attention.comm.is_hierarchical_comm_enable()
            else self.nccl_group,
            dist_attn_config=dist_attn_config,
        )
        mgr = dist_attn_runtime_dict_mgr[key]
        if mgr.dist_attn_runtime.overlap_degree == 0:
            self.skipTest("This config did not produce remote overlap stages")

        x = torch.randn(total_seqlen, head_dim, device=self.device, dtype=dtype)
        local_x = dispatch(x, key=key)
        sq = local_x.shape[0]
        local_q = torch.randn(sq, num_heads_q, head_dim, device=self.device, dtype=dtype)
        local_k = torch.randn(sq, num_heads_kv, head_dim, device=self.device, dtype=dtype)
        local_v = torch.randn(sq, num_heads_kv, head_dim, device=self.device, dtype=dtype)
        custom = CalcAttnCustomAttribute(
            return_block_max=True,
            return_block_lse=True,
            k_sparse_block_size=k_sparse_block_size,
        )
        _, meta = calc_attn(
            local_q, local_k, local_v, key,
            custom_attribute=custom,
        )

        # total n_chunks across all stages (host + remote)
        calc_meta = mgr.dist_attn_runtime.calc_meta
        total_sk = calc_meta.seqlen_k_local + sum(calc_meta.seqlen_k_per_remote_stage)
        n_chunks = (total_sk + k_sparse_block_size - 1) // k_sparse_block_size

        assert meta.block_max is not None, "block_max should not be None with overlap"
        assert meta.block_lse is not None, "block_lse should not be None with overlap"
        assert meta.block_max.shape == (sq, num_heads_q, n_chunks), (
            f"block_max shape mismatch: {meta.block_max.shape} vs expected ({sq}, {num_heads_q}, {n_chunks})"
        )
        assert meta.block_lse.shape == (sq, num_heads_q, n_chunks), (
            f"block_lse shape mismatch: {meta.block_lse.shape} vs expected ({sq}, {num_heads_q}, {n_chunks})"
        )
        assert meta.block_max.dtype == torch.float32
        assert meta.block_lse.dtype == torch.float32
        # at least some values should be finite (not all -inf)
        assert (meta.block_max > float("-inf")).any(), "block_max is all -inf"
        assert (meta.block_lse > float("-inf")).any(), "block_lse is all -inf"

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_calc_attn_block_meta_overlap_consistency(self):
        """block_max / block_lse from dist_attn (CP=2, overlap_degree > 0),
        after undispatch, must match torch reference for multi-doc varlen causal."""
        if not magi_attention.is_fa4_backend_enable():
            self.skipTest("FA4 backend not enabled")
        from magi_attention.functional import fa4 as fa4_mod

        if not fa4_mod.is_fa4_installed:
            self.skipTest("flash_attn not installed")

        configs = [
            # (total_seqlen, num_docs, num_heads_q, num_heads_kv)
            (1024,  1, 4, 4),
            (2048,  4, 4, 4),
            (4096,  8, 4, 4),
            (8192,  4, 4, 1),
        ]
        for i, (seqlen, ndocs, nhq, nhkv) in enumerate(configs):
            self._check_block_score_overlap(
                total_seqlen=seqlen, num_docs=ndocs,
                num_heads_q=nhq, num_heads_kv=nhkv, seed=42 + i,
            )

    def _check_block_score_overlap(
        self, total_seqlen, num_docs, num_heads_q, num_heads_kv,
        head_dim=64, block_size_k=128, chunk_size=128, seed=42,
    ):
        """Run one block_max / block_lse correctness check for CP>1 overlap."""
        import math, gc

        dtype = torch.bfloat16
        sm_scale = head_dim ** -0.5
        sm_scale_log2e = sm_scale * math.log2(math.e)

        torch.manual_seed(seed)
        if num_docs == 1:
            cu_seqlens = torch.tensor(
                [0, total_seqlen], dtype=torch.int32, device=self.device
            )
        else:
            raw = torch.randint(
                max(1, total_seqlen // (num_docs * 2)),
                total_seqlen // num_docs + 1,
                (num_docs,),
            )
            doc_lens = (raw.float() / raw.sum() * total_seqlen).int()
            doc_lens[-1] = total_seqlen - doc_lens[:-1].sum()
            cu_seqlens = torch.zeros(
                num_docs + 1, dtype=torch.int32, device=self.device
            )
            cu_seqlens[1:] = torch.cumsum(doc_lens.to(self.device), dim=0)
        max_seqlen = int(
            (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        )

        pad_size = compute_pad_size(total_seqlen, self.world_size, chunk_size)

        cost_kwargs = dict(
            calc_cost_factor=get_calc_cost_factor(
                num_heads_q=num_heads_q,
                head_dim=head_dim,
                tflops=H100_TFLOPS_16,
                mfu=H100_MATMUL_MFU,
            ),
            comm_cost_factor=get_comm_cost_factor(
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                bandwidth=H100_NVLINK_BANDWIDTH,
                bwu=H100_NVLINK_A2A_BWU,
                corr_factor=get_a2a_corr_factor(self.world_size),
            ),
        )

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=SequentialDispatchAlg()),
            overlap_config=OverlapConfig(
                enable=True,
                mode=AttnOverlapMode.STATIC,
                degree=4,
                min_chunk_size=block_size_k,
                max_num_chunks=64,
                alg=UniformOverlapAlg(random_costs=False),
                **cost_kwargs,
            ),
        )

        cp_group_or_mesh = (
            self.device_mesh
            if magi_attention.comm.is_hierarchical_comm_enable()
            else self.nccl_group
        )
        key = magi_attn_varlen_key(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=cp_group_or_mesh,
            causal=True,
            dist_attn_config=dist_attn_config,
        )
        mgr = dist_attn_runtime_dict_mgr[key]
        if mgr.dist_attn_runtime.overlap_degree == 0:
            return  # skip this config silently

        # All ranks generate the same full Q/K/V, then dispatch
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 1)
        full_q = torch.randn(
            total_seqlen, num_heads_q, head_dim,
            device=self.device, dtype=dtype,
        )
        full_k = torch.randn(
            total_seqlen, num_heads_kv, head_dim,
            device=self.device, dtype=dtype,
        )
        full_v = torch.randn(
            total_seqlen, num_heads_kv, head_dim,
            device=self.device, dtype=dtype,
        )

        local_q = dispatch(full_q, key=key)
        local_k = dispatch(full_k, key=key)
        local_v = dispatch(full_v, key=key)

        custom = CalcAttnCustomAttribute(
            return_block_max=True,
            return_block_lse=True,
            k_sparse_block_size=block_size_k,
        )

        _, meta = calc_attn(local_q, local_k, local_v, key, custom_attribute=custom)
        assert meta.block_max is not None and meta.block_lse is not None

        global_block_max = undispatch(meta.block_max, key=key)
        global_block_lse = undispatch(meta.block_lse, key=key)

        # --- PyTorch reference (per-doc causal) ---
        n_kblocks = math.ceil(max_seqlen / block_size_k)
        ref_block_max = torch.full(
            (total_seqlen, num_heads_q, n_kblocks),
            float("-inf"), dtype=torch.float32, device=self.device,
        )
        ref_block_lse = torch.full(
            (total_seqlen, num_heads_q, n_kblocks),
            float("-inf"), dtype=torch.float32, device=self.device,
        )

        gqa_ratio = num_heads_q // num_heads_kv
        for d in range(num_docs):
            qs = cu_seqlens[d].item()
            qe = cu_seqlens[d + 1].item()
            doc_len = qe - qs
            q_doc = full_q[qs:qe].float()
            k_doc = full_k[qs:qe].float()
            if gqa_ratio > 1:
                k_doc = k_doc.repeat_interleave(gqa_ratio, dim=1)
            q_idx = torch.arange(doc_len, device=self.device)

            n_doc_blocks = math.ceil(doc_len / block_size_k)
            for b in range(n_doc_blocks):
                k_s = b * block_size_k
                k_e = min((b + 1) * block_size_k, doc_len)
                qk_block = torch.einsum(
                    'qhd,khd->hqk', q_doc, k_doc[k_s:k_e]
                )
                causal_mask = q_idx[:, None] < torch.arange(
                    k_s, k_e, device=self.device
                )[None, :]
                qk_block[:, causal_mask] = float("-inf")
                ref_block_max[qs:qe, :, b] = (
                    qk_block.max(dim=2).values.T * sm_scale_log2e
                )
                ref_block_lse[qs:qe, :, b] = torch.logsumexp(
                    qk_block * sm_scale, dim=2
                ).T

        # --- compare ---
        n = min(n_kblocks, global_block_max.shape[2])
        label = f"{total_seqlen // 1024}K-{num_docs}doc-{num_heads_q}h"
        doc_lens_list = [int(cu_seqlens[i+1] - cu_seqlens[i]) for i in range(num_docs)]
        print(f"[_check_block_score_overlap] {label} | ws={self.world_size} "
              f"| doc_lens={doc_lens_list} | n_kblocks={n_kblocks}")

        for name, magi_out, ref_out in [
            ("block_max", global_block_max, ref_block_max),
            ("block_lse", global_block_lse, ref_block_lse),
        ]:
            magi_t = magi_out[:total_seqlen, :, :n]
            ref_t = ref_out[:, :, :n]

            ref_inf = ref_t == float("-inf")
            magi_inf = magi_t == float("-inf")
            disagree = (ref_inf != magi_inf).sum().item()
            assert disagree == 0, (
                f"[{label}] {name}: {disagree} -inf positions disagree"
            )

            valid = torch.isfinite(ref_t) & torch.isfinite(magi_t)
            assert valid.any(), f"[{label}] {name}: no valid entries"
            ref_v = ref_t[valid].double()
            magi_v = magi_t[valid].double()

            cos_sim = torch.nn.functional.cosine_similarity(
                ref_v.unsqueeze(0), magi_v.unsqueeze(0)
            ).item()
            print(f"  {name}: cos_sim={cos_sim:.10f}")
            assert cos_sim > 0.99999, (
                f"[{label}] {name} cosine similarity {cos_sim:.7f} < 0.99999"
            )
            torch.testing.assert_close(magi_v, ref_v, atol=1e-4, rtol=1e-4)

        del meta, ref_block_max, ref_block_lse
        gc.collect()
        torch.cuda.empty_cache()


class TestInterfaceWithWorldSize3(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize4(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize5(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize6(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize7(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize8(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)

    @skip_if_lt_x_gpu(8)
    @with_comms
    @switch_deterministic_mode_decorator(enable=True)
    def test_compiled_magiattn(self):
        # -----    skip for fa4 backend   ---- #

        if magi_attention.is_fa4_backend_enable():
            # TODO: support torch.compile and deterministic mode for fa4 backend
            return

        # --- Define attention config --- #

        total_seqlen = 32 * 1024  # 32k tokens
        num_heads_q = 48  # number of attention (query) heads
        num_heads_kv = 8  # number of key/value heads (GQA)
        head_dim = 128  # dimension of each attention head
        dtype = torch.bfloat16  # attention activation / computation dtype
        chunk_size = 512  # chunk size
        embed_dim = 4096  # token embedding tensor

        # --- Initialize MagiAttention meta configs for customized attention mask --- #

        q_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],  # 0~4k
                [4096, 8192],  # 4k~8k
                [8192, 12288],  # 8k~12k
                [12288, 16384],  # 12k~16k
                [16384, 20480],  # 16k~20k
                [20480, 24576],  # 20k~24k
                [24576, 28672],  # 24k~28k
                [28672, 32768],  # 28k~32k
            ]
        )
        k_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],  # 0~4k
                [0, 8192],  # 0~8k
                [0, 12288],  # 0~12k
                [0, 16384],  # 0~16k
                [0, 20480],  # 0~20k
                [0, 24576],  # 0~24k
                [0, 28672],  # 0~28k
                [0, 32768],  # 0~32k
            ]
        )
        attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)
        total_seqlen_q = total_seqlen_k = total_seqlen
        pad_size = compute_pad_size(  # pad embeds along seqlen dim for better performance
            total_seqlen_q=total_seqlen_q,
            cp_size=self.world_size,  # assuming we only have 1-dim context parallelism (cp)
            chunk_size=chunk_size,
        )

        global_dout = torch.randn(
            total_seqlen, num_heads_q, head_dim, device=self.device, dtype=dtype
        )
        dist.all_reduce(global_dout, group=self.nccl_group)

        q_proj = nn.Linear(
            embed_dim, num_heads_q * head_dim, dtype=dtype, device=self.device
        )
        k_proj = nn.Linear(
            embed_dim, num_heads_kv * head_dim, dtype=dtype, device=self.device
        )
        v_proj = nn.Linear(
            embed_dim, num_heads_kv * head_dim, dtype=dtype, device=self.device
        )

        # --- Compute magi_attn runtime key --- #

        magi_attn_runtime_key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=self.nccl_group,  # assuming we only have 1-dim context parallelism (cp)
        )

        total_out_ref, dx_ref = None, None
        for iter in range(6):
            use_compiled_magiattn = iter % 2 == 1

            torch.manual_seed(self.seed + iter // 2)
            x = torch.randn(
                total_seqlen,
                embed_dim,
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            dist.all_reduce(x.data, group=self.nccl_group)

            # --- Dispatch and pad --- #

            local_x = dispatch(x, key=magi_attn_runtime_key)

            # --- Simulate QKV projection --- #

            local_q = q_proj(local_x).view(-1, num_heads_q, head_dim)
            local_k = k_proj(local_x).view(-1, num_heads_kv, head_dim)
            local_v = v_proj(local_x).view(-1, num_heads_kv, head_dim)

            # --- Apply compiled magi_attn func --- #

            # NOTE: since torch.compile does not support async dist comm,
            # we can not compile it with fullgraph=True
            magiattn_func = (
                torch.compile(fullgraph=False)(calc_attn)
                if use_compiled_magiattn
                else calc_attn
            )
            local_out, _ = magiattn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                key=magi_attn_runtime_key,
            )

            # --- Undispatch and unpad --- #

            total_out = undispatch(
                x=local_out,
                key=magi_attn_runtime_key,
            )

            total_out.backward(global_dout)

            dx = x.grad

            if use_compiled_magiattn:
                assert total_out_ref is not None and dx_ref is not None
                torch.testing.assert_close(total_out, total_out_ref)
                torch.testing.assert_close(dx, dx_ref)
            else:
                total_out_ref = total_out.detach().clone()
                dx_ref = dx.detach().clone()


if __name__ == "__main__":
    run_tests()
