# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Exact FlashRWKV-backed public RWKV7 recurrent API."""

import torch

from fla.ops.backends import dispatch
from fla.ops.cp import FLACPContext
from fla.ops.rwkv7.backends.provider import (
    set_last_rwkv7_kernel,
    set_last_rwkv7_provider,
)


@dispatch("rwkv7")
def recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    state_indices: torch.LongTensor | None = None,
    mode: str = "fp32io16",
    safe_gate: bool = False,
    chunk_size: int | None = None,
    disable_recompute: bool = False,
    cp_context: FLACPContext | None = None,
    **kwargs,
):
    """Run the exact self-owned FlashRWKV recurrent provider.

    Fixed inputs with gradients use FlashRWKV recurrent autograd. Ordinary
    fixed or packed forward calls use ``algorithm='recurrent'``. Packed serving
    with ``state_indices`` updates the supplied state pool in place and returns
    that same object as final state.

    This public boundary has no Triton, chunk, or reference fallback.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    set_last_rwkv7_provider(None)
    set_last_rwkv7_kernel(None)
    raise RuntimeError(
        "public RWKV7 recurrent execution requires the exact FlashRWKV "
        "backend; fallback is disabled"
    )


__all__ = ["recurrent_rwkv7"]
