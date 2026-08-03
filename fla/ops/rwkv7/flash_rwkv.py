# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Public access to the exact, self-owned FlashRWKV operator suite.

This provider namespace preserves FlashRWKV's public signatures while owning
optional-dependency admission, exact source provenance, and per-call telemetry
inside FLA. It intentionally does not expose FlashRWKV's private ``_C`` launch
variants.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from fla.ops.rwkv7.backends.flash_rwkv import (
    FLASH_RWKV_PUBLIC_OPERATORS,
    FLASH_RWKV_REQUIRED_OPERATORS,
    preflight_flash_rwkv_installation,
)
from fla.ops.rwkv7.backends.provider import (
    set_last_rwkv7_kernel,
    set_last_rwkv7_provider,
)

if TYPE_CHECKING:
    from flash_rwkv import ChunkConfig


def _invoke(operator: str, *args, **kwargs) -> Any:
    if operator not in FLASH_RWKV_REQUIRED_OPERATORS:
        raise RuntimeError(f"unregistered FlashRWKV operator: {operator}")
    set_last_rwkv7_provider(None)
    set_last_rwkv7_kernel(None)
    preflight_flash_rwkv_installation()
    provider = importlib.import_module("flash_rwkv")
    result = getattr(provider, operator)(*args, **kwargs)
    set_last_rwkv7_provider("flash_rwkv")
    set_last_rwkv7_kernel(operator)
    return result


def pretrain_recurrent_fp32io16_forward(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
):
    return _invoke(
        "pretrain_recurrent_fp32io16_forward",
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
    )


def rwkv7_recurrent(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    mode: str = "fp32io16",
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
):
    return _invoke(
        "rwkv7_recurrent",
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


def rwkv7_recurrent_stateful(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fp32io16",
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
):
    return _invoke(
        "rwkv7_recurrent_stateful",
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        scale=scale,
        mode=mode,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )


def prepare_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    total_tokens: int,
    state_pool_size: int,
):
    return _invoke(
        "prepare_recurrent_metadata",
        cu_seqlens,
        state_indices,
        total_tokens=total_tokens,
        state_pool_size=state_pool_size,
    )


def rwkv7(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    mode: str = "fp32io16",
    algorithm: str = "auto",
    chunk_size: int | None = None,
    chunk_config: ChunkConfig | None = None,
    decay_bias: torch.Tensor | None = None,
    elapsed_t: torch.Tensor | None = None,
    validated_metadata: object | None = None,
):
    return _invoke(
        "rwkv7",
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
        algorithm=algorithm,
        chunk_size=chunk_size,
        chunk_config=chunk_config,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
    )


def pretrain_tmix_a_gate_bf16(
    a0: torch.Tensor,
    a12: torch.Tensor,
):
    return _invoke("pretrain_tmix_a_gate_bf16", a0, a12)


def pretrain_tmix_vres_gate_bf16(
    value: torch.Tensor,
    first_value: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
):
    return _invoke(
        "pretrain_tmix_vres_gate_bf16",
        value,
        first_value,
        v0,
        v12,
    )


def pretrain_tmix_mix6_bf16(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
):
    return _invoke(
        "pretrain_tmix_mix6_bf16",
        x,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
    )


def pretrain_tmix_kk_pre_bf16(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    learning_rate: torch.Tensor,
    learning_rate_scale: torch.Tensor,
):
    return _invoke(
        "pretrain_tmix_kk_pre_bf16",
        key,
        key_scale,
        learning_rate,
        learning_rate_scale,
    )


def pretrain_tmix_lnx_rkvres_xg_bf16(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
):
    return _invoke(
        "pretrain_tmix_lnx_rkvres_xg_bf16",
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )


def pretrain_cmix_bf16(
    x: torch.Tensor,
    x_k: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
):
    return _invoke(
        "pretrain_cmix_bf16",
        x,
        x_k,
        key_weight,
        value_weight,
    )


def pretrain_l2wrap_ce_bf16(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    return _invoke("pretrain_l2wrap_ce_bf16", logits, targets)


def pretrain_head_l2wrap_ce_bf16(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_rows: int = 4096,
):
    return _invoke(
        "pretrain_head_l2wrap_ce_bf16",
        hidden,
        weight,
        targets,
        chunk_rows=chunk_rows,
    )


def infer_tmix_mix6_fp16(
    x: torch.Tensor,
    shift_state: torch.Tensor,
    mixes: Sequence[torch.Tensor],
):
    return _invoke("infer_tmix_mix6_fp16", x, shift_state, mixes)


def infer_tmix_kk_a_gate_fp16(
    key: torch.Tensor,
    key_scale: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_delta: torch.Tensor,
    key_gate_scale: torch.Tensor,
):
    return _invoke(
        "infer_tmix_kk_a_gate_fp16",
        key,
        key_scale,
        gate_bias,
        gate_delta,
        key_gate_scale,
    )


def infer_tmix_lnx_rkvres_xg_fp16(
    recurrent_output: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    gate: torch.Tensor,
):
    return _invoke(
        "infer_tmix_lnx_rkvres_xg_fp16",
        recurrent_output,
        receptance,
        key,
        value,
        residual_scale,
        norm_weight,
        norm_bias,
        gate,
    )


def infer_tmix_vres_gate_fp16(
    value: torch.Tensor,
    first_value: torch.Tensor,
    gate_bias: torch.Tensor,
    gate_delta: torch.Tensor,
):
    return _invoke(
        "infer_tmix_vres_gate_fp16",
        value,
        first_value,
        gate_bias,
        gate_delta,
    )


def infer_cmix_mix_fp16(
    x: torch.Tensor,
    shift_state: torch.Tensor,
    mix: torch.Tensor,
):
    return _invoke("infer_cmix_mix_fp16", x, shift_state, mix)


__all__ = list(FLASH_RWKV_PUBLIC_OPERATORS)
