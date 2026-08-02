# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit FLA chunk oracle retained only for tests and comparisons."""

import torch

from fla.ops.cp import FLACPContext
from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule


def chunk_rwkv7_reference(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    chunk_size: int | None = None,
    disable_recompute: bool = False,
    cp_context: FLACPContext | None = None,
):
    """Run the upstream FLA chunk implementation as an explicit oracle."""
    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        safe_gate=safe_gate,
        chunk_size=chunk_size,
        disable_recompute=disable_recompute,
        cp_context=cp_context,
    )


__all__ = ["chunk_rwkv7_reference"]
