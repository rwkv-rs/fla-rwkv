# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Optional FlashRWKV backend for RWKV7 chunk execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fla.ops.backends import BaseBackend
from fla.ops.rwkv7.backends.provider import set_last_rwkv7_provider

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext


class FlashRWKVBackend(BaseBackend):
    """FlashRWKV FP32-state chunk backend."""

    backend_type = 'flash_rwkv'
    package_name = 'flash_rwkv'
    env_var = 'FLA_FLASH_RWKV'
    default_enable = False
    priority = 3

    def chunk_rwkv7_verifier(
        self,
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
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        del scale, output_final_state, cu_seqlens_cpu, safe_gate
        tensors = (r, w, k, v, a, b)
        if not all(tensor.is_cuda for tensor in tensors):
            return False, 'FlashRWKV requires CUDA tensors'
        if any(tensor.dtype != torch.float16 for tensor in tensors):
            return False, 'FlashRWKV requires float16 inputs'
        if any(tensor.device != r.device for tensor in tensors):
            return False, 'FlashRWKV requires all inputs on the same CUDA device'
        if any(tensor.shape != r.shape for tensor in (w, k, a, b)) or v.shape[:3] != r.shape[:3]:
            return False, 'FlashRWKV requires matching [B, T, H, D] input shapes'
        if r.shape[-1] != 64 or v.shape[-1] != 64:
            return False, f'FlashRWKV requires K=V=64, got K={r.shape[-1]}, V={v.shape[-1]}'
        requires_grad = any(tensor.requires_grad for tensor in tensors) or (
            initial_state is not None and initial_state.requires_grad
        )
        if requires_grad and cu_seqlens is not None:
            return False, 'FlashRWKV packed execution is forward-only'
        if requires_grad and initial_state is not None and initial_state.dtype != torch.float32:
            return False, 'FlashRWKV training requires an FP32 initial_state'
        if initial_state is not None and initial_state.device != r.device:
            return False, 'FlashRWKV requires initial_state on the input device'
        if cp_context is not None:
            return False, 'FlashRWKV does not support context parallel execution'
        if disable_recompute:
            return False, 'FlashRWKV does not support disable_recompute=True'
        if chunk_size not in {None, 16, 32, 64}:
            return False, f'FlashRWKV chunk_size must be 16, 32, or 64, got {chunk_size}'
        if kwargs:
            return False, f'FlashRWKV does not support extra arguments: {sorted(kwargs)}'
        return True, None

    def chunk_rwkv7(
        self,
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
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        del cu_seqlens_cpu, safe_gate, disable_recompute, cp_context, kwargs
        import flash_rwkv

        output = flash_rwkv.rwkv7(
            r,
            w,
            k,
            v,
            a,
            b,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            mode='fp32io16',
            algorithm='chunk',
            chunk_size=chunk_size,
        )
        set_last_rwkv7_provider('flash_rwkv')
        return output


__all__ = ['FlashRWKVBackend']
