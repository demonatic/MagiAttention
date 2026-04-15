#!/usr/bin/env bash
# Run calc_attn block_max / block_lse related tests with FA4 backend enabled.
# Requires: installed flash_attn (cute / FA4), GPU, and for WS2 test at least 2 GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export MAGI_ATTENTION_FA4_BACKEND=1
exec python -m pytest \
  tests/test_api/test_interface.py::TestInterfaceBaseWithWorldSize1::test_calc_attn_custom_attribute_fa4_block_shapes \
  tests/test_api/test_interface.py::TestInterfaceBaseWithWorldSize1::test_calc_attn_custom_attribute_non_fa4_raises \
  tests/test_api/test_interface.py::TestInterfaceWithWorldSize2::test_calc_attn_custom_attribute_overlap_block_shapes \
  tests/test_api/test_interface.py::TestInterfaceWithWorldSize2::test_calc_attn_block_meta_overlap_consistency \
  -v --tb=short "$@"
