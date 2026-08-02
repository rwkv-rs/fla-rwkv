# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Eligibility shared by ordinary RWKV7 FlashRWKV inference call sites."""

from __future__ import annotations

import torch


def can_use_flash_rwkv_inference(
    *tensors: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    head_dim: int | None = None,
) -> bool:
    """Return whether fused FlashRWKV FP16 inference preserves this contract."""

    if torch.is_grad_enabled() or cu_seqlens is not None:
        return False
    if head_dim is not None and head_dim != 64:
        return False
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    reference = tensors[0]
    return all(
        tensor.is_cuda
        and tensor.dtype == torch.float16
        and tensor.device == reference.device
        and tensor.is_contiguous()
        for tensor in tensors
    )


__all__ = ["can_use_flash_rwkv_inference"]
