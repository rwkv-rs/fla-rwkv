# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Optional FlashRWKV backend for RWKV7 chunk execution."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import math
import re
import subprocess
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import torch

from fla.ops.backends import BaseBackend
from fla.ops.rwkv7.backends.provider import set_last_rwkv7_provider

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext

FLASH_RWKV_SOURCE_REVISION = "866aafd2eed146b0eda1ce03444009ae030f89e3"


@cache
def _installed_flash_rwkv_revision() -> str | None:
    try:
        distribution = importlib.metadata.distribution("flash-rwkv")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    revision = metadata.get("vcs_info", {}).get("commit_id")
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    source = metadata.get("url")
    if not isinstance(source, str):
        return None
    parsed = urlparse(source)
    if parsed.scheme != "file":
        return None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(Path(unquote(parsed.path))), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


class FlashRWKVBackend(BaseBackend):
    """FlashRWKV FP32-state chunk backend."""

    backend_type = "flash_rwkv"
    package_name = "flash_rwkv"
    env_var = "FLA_FLASH_RWKV"
    default_enable = False
    fail_closed_on_explicit_enable = True
    priority = 3

    def on_explicit_failure(self) -> None:
        set_last_rwkv7_provider(None)

    @classmethod
    def is_available(cls) -> bool:
        try:
            module = importlib.import_module(cls.package_name)
            version_match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", getattr(module, "__version__", ""))
            if version_match is None or tuple(map(int, version_match.groups())) < (0, 1, 0):
                return False
            parameters = inspect.signature(module.rwkv7).parameters
            stateful_parameters = inspect.signature(module.rwkv7_recurrent_stateful).parameters
        except (AttributeError, ImportError, TypeError, ValueError):
            return False
        required_parameters = {
            "r",
            "log_decay",
            "k",
            "v",
            "a",
            "b",
            "scale",
            "initial_state",
            "output_final_state",
            "cu_seqlens",
            "mode",
            "algorithm",
            "chunk_size",
        }
        required_stateful_parameters = {
            "r",
            "log_decay",
            "k",
            "v",
            "a",
            "b",
            "state_pool",
            "cu_seqlens",
            "state_indices",
            "scale",
            "mode",
        }
        return (
            required_parameters <= parameters.keys()
            and required_stateful_parameters <= stateful_parameters.keys()
            and _installed_flash_rwkv_revision() == FLASH_RWKV_SOURCE_REVISION
        )

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
        state_indices: torch.LongTensor | None = None,
        mode: str = "fp32io16",
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ) -> tuple[bool, str | None]:
        del output_final_state
        tensors = (r, w, k, v, a, b)
        if any(getattr(tensor, "ndim", None) != 4 for tensor in tensors):
            return False, "FlashRWKV requires rank-4 [B, T, H, D] inputs"
        if any(not tensor.is_contiguous() for tensor in tensors):
            return False, "FlashRWKV requires contiguous inputs"
        if not all(tensor.is_cuda for tensor in tensors):
            return False, "FlashRWKV requires CUDA tensors"
        if any(tensor.dtype not in {torch.float16, torch.bfloat16} for tensor in tensors):
            return False, "FlashRWKV requires float16 or bfloat16 inputs"
        if any(tensor.device != r.device for tensor in tensors):
            return False, "FlashRWKV requires all inputs on the same CUDA device"
        if any(tensor.shape != r.shape for tensor in (w, k, a, b)) or v.shape[:3] != r.shape[:3]:
            return False, "FlashRWKV requires matching [B, T, H, D] input shapes"
        if any(dimension <= 0 for dimension in r.shape) or v.shape[-1] <= 0:
            return False, "FlashRWKV requires positive B, T, H, K, and V dimensions"
        if r.shape[-1] != 64 or v.shape[-1] != 64:
            return False, f"FlashRWKV requires K=V=64, got K={r.shape[-1]}, V={v.shape[-1]}"
        try:
            if not math.isfinite(float(scale)):
                return False, "FlashRWKV requires a finite scale"
        except (TypeError, ValueError):
            return False, "FlashRWKV requires a finite scale"
        if mode not in {"fp32io16", "fp16"}:
            return False, "FlashRWKV mode must be 'fp32io16' or 'fp16'"
        if mode == "fp16" and any(tensor.dtype != torch.float16 for tensor in tensors):
            return False, "FlashRWKV mode='fp16' requires float16 inputs"
        requires_grad = any(tensor.requires_grad for tensor in tensors) or (
            initial_state is not None and initial_state.requires_grad
        )
        if requires_grad and cu_seqlens is not None:
            return False, "FlashRWKV packed execution is forward-only"
        expected_state_rows = r.shape[0]
        if cu_seqlens is not None:
            if r.shape[0] != 1:
                return False, "FlashRWKV packed execution requires B=1"
            if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in {torch.int32, torch.int64}:
                return False, "FlashRWKV cu_seqlens must be a rank-1 int32 or int64 tensor"
            if not cu_seqlens.is_contiguous():
                return False, "FlashRWKV cu_seqlens must be contiguous"
            if cu_seqlens.device.type != "cpu" and cu_seqlens.device != r.device:
                return False, "FlashRWKV cu_seqlens must be on CPU or the input device"
            offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
            if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != r.shape[1]:
                return False, "FlashRWKV cu_seqlens must span the packed token count from 0"
            if any(end <= start for start, end in zip(offsets[:-1], offsets[1:], strict=True)):
                return False, "FlashRWKV cu_seqlens must be strictly increasing"
            expected_state_rows = len(offsets) - 1
            if state_indices is not None:
                if initial_state is None:
                    return False, "FlashRWKV state_indices requires an initial state pool"
                if state_indices.ndim != 1 or state_indices.dtype not in {torch.int32, torch.int64}:
                    return False, "FlashRWKV state_indices must be a rank-1 int32 or int64 tensor"
                if not state_indices.is_contiguous():
                    return False, "FlashRWKV state_indices must be contiguous"
                if state_indices.device.type != "cpu" and state_indices.device != r.device:
                    return False, "FlashRWKV state_indices must be on CPU or the input device"
                indices = tuple(int(value) for value in state_indices.detach().cpu().tolist())
                if len(indices) != expected_state_rows:
                    return False, "FlashRWKV state_indices length must match the packed sequence count"
                if len(set(indices)) != len(indices):
                    return False, "FlashRWKV state_indices must be unique within one call"
            if cu_seqlens_cpu is not None:
                if (
                    cu_seqlens_cpu.device.type != "cpu"
                    or cu_seqlens_cpu.ndim != 1
                    or cu_seqlens_cpu.dtype not in {torch.int32, torch.int64}
                    or not cu_seqlens_cpu.is_contiguous()
                ):
                    return False, "FlashRWKV cu_seqlens_cpu must be a contiguous rank-1 CPU integer tensor"
                if tuple(int(value) for value in cu_seqlens_cpu.tolist()) != offsets:
                    return False, "FlashRWKV cu_seqlens_cpu must match cu_seqlens"
        elif cu_seqlens_cpu is not None:
            return False, "FlashRWKV cu_seqlens_cpu requires cu_seqlens"
        elif state_indices is not None:
            return False, "FlashRWKV state_indices requires cu_seqlens"
        if initial_state is not None:
            expected_trailing_shape = (r.shape[2], r.shape[3], v.shape[3])
            if (
                initial_state.ndim != 4
                or initial_state.shape[1:] != expected_trailing_shape
                or (state_indices is None and initial_state.shape[0] != expected_state_rows)
            ):
                return False, "FlashRWKV initial_state must have shape [N, H, K, V] matching the input layout"
            if not initial_state.is_floating_point():
                return False, "FlashRWKV initial_state must have a floating-point dtype"
            if not initial_state.is_contiguous():
                return False, "FlashRWKV initial_state must be contiguous"
            if initial_state.device != r.device:
                return False, "FlashRWKV requires initial_state on the input device"
            if requires_grad and initial_state.dtype != torch.float32:
                return False, "FlashRWKV training requires an FP32 initial_state"
            if state_indices is not None:
                if any(index < 0 or index >= initial_state.shape[0] for index in indices):
                    return False, "FlashRWKV state_indices entries must be within the state pool"
                expected_state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
                if initial_state.dtype != expected_state_dtype:
                    return False, f"FlashRWKV stateful {mode} requires a {expected_state_dtype} state pool"
        if cp_context is not None:
            return False, "FlashRWKV does not support context parallel execution"
        if safe_gate:
            return False, "FlashRWKV does not expose the FLA safe_gate contract"
        if disable_recompute:
            return False, "FlashRWKV does not support disable_recompute=True"
        if chunk_size not in {None, 16, 32, 64}:
            return False, f"FlashRWKV chunk_size must be 16, 32, or 64, got {chunk_size}"
        if kwargs:
            return False, f"FlashRWKV does not support extra arguments: {sorted(kwargs)}"
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
        state_indices: torch.LongTensor | None = None,
        mode: str = "fp32io16",
        safe_gate: bool = False,
        chunk_size: int | None = None,
        disable_recompute: bool = False,
        cp_context: FLACPContext | None = None,
        **kwargs,
    ):
        del cu_seqlens_cpu, safe_gate, disable_recompute, cp_context, kwargs
        import flash_rwkv

        set_last_rwkv7_provider(None)
        if state_indices is not None:
            output = flash_rwkv.rwkv7_recurrent_stateful(
                r,
                w,
                k,
                v,
                a,
                b,
                state_pool=initial_state,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                scale=scale,
                mode=mode,
            )
            output = (output, initial_state if output_final_state else None)
        else:
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
                mode=mode,
                algorithm="chunk",
                chunk_size=chunk_size,
            )
        set_last_rwkv7_provider("flash_rwkv")
        return output


__all__ = ["FLASH_RWKV_SOURCE_REVISION", "FlashRWKVBackend"]
