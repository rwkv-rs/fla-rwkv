# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Context-parallel coverage for the explicitly selected RWKV7 chunk oracle."""

import pytest
import torch

# tests/context_parallel is on sys.path under pytest (see pyproject pythonpath)
from test_cp_dplr import run_cp_test_with_spawn

from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule


def _explicit_chunk_oracle(q, k, v, a, b, gk, scale=None, **kwargs):
    return chunk_dplr_delta_rule(
        q=q,
        gk=gk,
        k=k,
        v=v,
        a=a,
        b=b,
        scale=scale,
        **kwargs,
    )


def test_cp2_explicit_oracle_sequence_cut():
    """CP2 with sequences cut across the rank boundary."""
    if torch.cuda.device_count() < 2:
        pytest.skip("At least 2 GPUs required")
    run_cp_test_with_spawn(
        world_size=2,
        test_name="CP2_ExplicitOracle_SequenceCut",
        T=1024,
        H=4,
        D=64,
        lengths=[400, 624],
        op=_explicit_chunk_oracle,
    )


def test_cp4_explicit_oracle_single_sequence():
    """CP4 with one long sequence spanning all ranks."""
    if torch.cuda.device_count() < 4:
        pytest.skip("At least 4 GPUs required")
    run_cp_test_with_spawn(
        world_size=4,
        test_name="CP4_ExplicitOracle_SingleSequence",
        T=1024,
        H=4,
        D=64,
        lengths=[1024],
        op=_explicit_chunk_oracle,
    )
