# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Exact FlashRWKV-backed public RWKV7 recurrent APIs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from fla.ops.rwkv7.backends import rwkv7_flash_backend
from fla.ops.rwkv7.backends.flash_rwkv import FlashRWKVProvenanceError
from fla.ops.rwkv7.backends.provider import (
    set_last_rwkv7_kernel,
    set_last_rwkv7_provider,
)

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext

_DISPATCH_DISABLED = os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1"
_FLASH_RWKV_DISABLED = os.environ.get("FLA_FLASH_RWKV", "1") == "0"


def _fail_closed(message: str, *, cause: Exception | None = None) -> None:
    set_last_rwkv7_provider(None)
    set_last_rwkv7_kernel(None)
    error = RuntimeError(f"public RWKV7 recurrent execution {message}; fallback is disabled")
    if cause is None:
        raise error
    raise error from cause


def prepare_rwkv7_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    total_tokens: int,
    state_pool_size: int,
) -> object:
    """Validate packed metadata once and return FlashRWKV's opaque native ticket."""

    if _DISPATCH_DISABLED:
        _fail_closed("is disabled by FLA_DISABLE_BACKEND_DISPATCH=1")
    if _FLASH_RWKV_DISABLED:
        _fail_closed("requires the enabled FlashRWKV backend")
    try:
        return rwkv7_flash_backend.prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=total_tokens,
            state_pool_size=state_pool_size,
        )
    except FlashRWKVProvenanceError as error:
        _fail_closed(f"failed FlashRWKV provenance validation: {error}", cause=error)


def recurrent_rwkv7(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
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
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
    **kwargs,
):
    """Run product RWKV7 recurrence from raw decay logits.

    This standard entry point never materializes canonical log-decay in Python.
    It validates and calls the exact FlashRWKV raw-decay provider directly,
    without general backend iteration. Fixed inputs with gradients select the
    native recurrent autograd path, ordinary fixed or packed inference selects
    the native functional path, and packed calls with ``state_indices`` update
    the supplied state pool in place and return that same object.
    """

    if _DISPATCH_DISABLED:
        _fail_closed("is disabled by FLA_DISABLE_BACKEND_DISPATCH=1")
    if _FLASH_RWKV_DISABLED:
        _fail_closed("requires the enabled FlashRWKV backend")
    if cu_seqlens_cpu is not None:
        _fail_closed("does not accept duplicate CPU packed metadata")
    if safe_gate or chunk_size is not None or disable_recompute or cp_context is not None or kwargs:
        _fail_closed("received unsupported chunk/context arguments")
    try:
        return rwkv7_flash_backend.recurrent_rwkv7(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
        )
    except FlashRWKVProvenanceError as error:
        _fail_closed(f"failed FlashRWKV provenance validation: {error}", cause=error)


__all__ = ["prepare_rwkv7_recurrent_metadata", "recurrent_rwkv7"]
