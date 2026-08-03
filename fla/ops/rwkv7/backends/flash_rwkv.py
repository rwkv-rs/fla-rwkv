# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Exact FlashRWKV backend for public RWKV7 recurrent execution."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import math
import re
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import torch

from fla.ops.backends import BaseBackend
from fla.ops.rwkv7.backends.provider import (
    set_last_rwkv7_kernel,
    set_last_rwkv7_provider,
)

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext

FLASH_RWKV_SOURCE_REVISION = "5410491f0d6cff6058e5bd21cbab900b5b54f220"
FLASH_RWKV_REPOSITORY = "https://github.com/rwkv-rs/FlashRWKV.git"
FLASH_RWKV_REQUIRED_OPERATORS = (
    "infer_cmix_mix_fp16",
    "infer_tmix_kk_a_gate_fp16",
    "infer_tmix_lnx_rkvres_xg_fp16",
    "infer_tmix_mix6_fp16",
    "infer_tmix_vres_gate_fp16",
    "pretrain_cmix_bf16",
    "pretrain_head_l2wrap_ce_bf16",
    "pretrain_l2wrap_ce_bf16",
    "pretrain_recurrent_fp32io16_forward",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "pretrain_tmix_mix6_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "prepare_recurrent_metadata",
    "rwkv7",
    "rwkv7_recurrent",
    "rwkv7_recurrent_stateful",
)
FLASH_RWKV_PUBLIC_OPERATORS = tuple(
    {
        "pretrain_recurrent_fp32io16_forward": "pretrain_recurrent_fp32io16_from_decay_logits",
        "rwkv7": "rwkv7_from_decay_logits",
        "rwkv7_recurrent": "rwkv7_recurrent_from_decay_logits",
        "rwkv7_recurrent_stateful": "rwkv7_recurrent_stateful_from_decay_logits",
    }.get(name, name)
    for name in FLASH_RWKV_REQUIRED_OPERATORS
)


class FlashRWKVProvenanceError(RuntimeError):
    """The installed FlashRWKV distribution does not own the imported provider."""


@dataclass(frozen=True)
class FlashRWKVProvenance:
    repository: str
    revision: str
    editable: bool
    distribution_root: Path
    module_path: Path
    native_extension_path: Path


def _read_direct_url(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        raise FlashRWKVProvenanceError("flash-rwkv lacks PEP 610 direct_url.json")
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise FlashRWKVProvenanceError("flash-rwkv direct_url.json is invalid") from error
    if not isinstance(payload, dict):
        raise FlashRWKVProvenanceError("flash-rwkv direct_url.json must be an object")
    return payload


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FlashRWKVProvenanceError(
            f"failed to inspect editable FlashRWKV checkout: git {' '.join(arguments)}"
        ) from error


def _canonical_repository(url: str) -> str | None:
    candidate = url.removeprefix("git+")
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in candidate):
        return None
    if candidate in {
        "git@github.com:rwkv-rs/FlashRWKV.git",
        "ssh://git@github.com/rwkv-rs/FlashRWKV.git",
    }:
        return FLASH_RWKV_REPOSITORY
    parsed = urlparse(candidate)
    try:
        repository_path = parsed.path.encode("ascii").decode("ascii").lower()
    except UnicodeEncodeError:
        return None
    if (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and repository_path
        in {
            "/rwkv-rs/flashrwkv",
            "/rwkv-rs/flashrwkv/",
            "/rwkv-rs/flashrwkv.git",
            "/rwkv-rs/flashrwkv.git/",
        }
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    ):
        return FLASH_RWKV_REPOSITORY
    return None


def _module_path(module: ModuleType, name: str) -> Path:
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        raise FlashRWKVProvenanceError(f"{name} has no concrete module file")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FlashRWKVProvenanceError(f"{name} module file does not exist: {resolved}")
    return resolved


def _validate_public_api(module: ModuleType) -> None:
    version_match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?",
        getattr(module, "__version__", ""),
    )
    if version_match is None or tuple(map(int, version_match.groups())) < (0, 1, 0):
        raise FlashRWKVProvenanceError("FlashRWKV version must be at least 0.1.0")
    missing_operators = tuple(
        name
        for name in FLASH_RWKV_REQUIRED_OPERATORS
        if not callable(getattr(module, name, None))
    )
    if missing_operators:
        raise FlashRWKVProvenanceError(
            "FlashRWKV public operator suite is incomplete: "
            f"{', '.join(missing_operators)}"
        )
    try:
        raw_parameters = inspect.signature(
            module.rwkv7_recurrent
        ).parameters
        raw_stateful_parameters = inspect.signature(
            module.rwkv7_recurrent_stateful
        ).parameters
        raw_training_parameters = inspect.signature(
            module.pretrain_recurrent_fp32io16_forward
        ).parameters
        prepare_metadata_parameters = inspect.signature(
            module.prepare_recurrent_metadata
        ).parameters
    except (AttributeError, TypeError, ValueError) as error:
        raise FlashRWKVProvenanceError("FlashRWKV public RWKV7 API is unavailable") from error
    required_raw_parameters = {
        "r",
        "decay_logits",
        "k",
        "v",
        "a",
        "b",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
        "decay_bias",
        "elapsed_t",
        "validated_metadata",
    }
    required_raw_stateful_parameters = {
        "r",
        "decay_logits",
        "k",
        "v",
        "a",
        "b",
        "state_pool",
        "cu_seqlens",
        "state_indices",
        "scale",
        "mode",
        "decay_bias",
        "elapsed_t",
        "validated_metadata",
    }
    required_raw_training_parameters = {
        "r",
        "decay_logits",
        "k",
        "v",
        "a",
        "b",
        "scale",
        "initial_state",
        "output_final_state",
        "decay_bias",
        "elapsed_t",
    }
    required_prepare_metadata_parameters = {
        "cu_seqlens",
        "state_indices",
        "total_tokens",
        "state_pool_size",
    }
    if not required_raw_parameters <= raw_parameters.keys():
        raise FlashRWKVProvenanceError(
            "FlashRWKV raw-decay recurrent signature is incompatible"
        )
    if not required_raw_stateful_parameters <= raw_stateful_parameters.keys():
        raise FlashRWKVProvenanceError(
            "FlashRWKV raw-decay stateful recurrent signature is incompatible"
        )
    if not required_raw_training_parameters <= raw_training_parameters.keys():
        raise FlashRWKVProvenanceError(
            "FlashRWKV raw-decay recurrent autograd signature is incompatible"
        )
    if not required_prepare_metadata_parameters <= prepare_metadata_parameters.keys():
        raise FlashRWKVProvenanceError(
            "FlashRWKV recurrent metadata preparation signature is incompatible"
        )
    if not callable(getattr(module, "validate_packed_metadata_strict", None)):
        raise FlashRWKVProvenanceError(
            "FlashRWKV strict packed metadata validator is unavailable"
        )


def validate_flash_rwkv_installation() -> FlashRWKVProvenance:
    """Validate exact distribution, Python module, and native extension ownership."""

    try:
        distribution = importlib.metadata.distribution("flash-rwkv")
        module = importlib.import_module("flash_rwkv")
        native_module = importlib.import_module("flash_rwkv._C")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise FlashRWKVProvenanceError("exact flash-rwkv distribution is unavailable") from error

    _validate_public_api(module)
    module_path = _module_path(module, "flash_rwkv")
    native_path = _module_path(native_module, "flash_rwkv._C")
    if getattr(module, "_C", None) is not native_module:
        raise FlashRWKVProvenanceError(
            "flash_rwkv._C is not the native module owned by the imported package"
        )
    if not any(
        str(native_path).endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise FlashRWKVProvenanceError("flash_rwkv._C is not a native extension")

    direct_url = _read_direct_url(distribution)
    source = direct_url.get("url")
    if not isinstance(source, str):
        raise FlashRWKVProvenanceError("flash-rwkv direct URL is missing")
    parsed = urlparse(source)
    editable = direct_url.get("dir_info", {}).get("editable") is True

    if editable:
        if parsed.scheme != "file":
            raise FlashRWKVProvenanceError("editable flash-rwkv must use a file URL")
        source_root = Path(unquote(parsed.path)).resolve()
        git_root = Path(_git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
        if git_root != source_root:
            raise FlashRWKVProvenanceError(
                "editable flash-rwkv direct URL is not its Git top-level"
            )
        origin = _git_output(source_root, "remote", "get-url", "origin")
        repository = _canonical_repository(origin)
        if repository is None:
            raise FlashRWKVProvenanceError(
                f"editable flash-rwkv origin is not {FLASH_RWKV_REPOSITORY}"
            )
        revision = _git_output(source_root, "rev-parse", "HEAD")
        if revision != FLASH_RWKV_SOURCE_REVISION:
            raise FlashRWKVProvenanceError(
                f"editable flash-rwkv HEAD must be {FLASH_RWKV_SOURCE_REVISION}"
            )
        status = _git_output(
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if status:
            raise FlashRWKVProvenanceError("editable flash-rwkv checkout is dirty")
        package_root = source_root / "flash_rwkv"
        distribution_root = source_root
    else:
        repository = _canonical_repository(source)
        if repository is None:
            raise FlashRWKVProvenanceError(
                f"flash-rwkv must come from {FLASH_RWKV_REPOSITORY}"
            )
        vcs_info = direct_url.get("vcs_info", {})
        revision = vcs_info.get("commit_id")
        if vcs_info.get("vcs") != "git" or revision != FLASH_RWKV_SOURCE_REVISION:
            raise FlashRWKVProvenanceError(
                f"flash-rwkv PEP 610 commit must be {FLASH_RWKV_SOURCE_REVISION}"
            )
        package_root = Path(distribution.locate_file("flash_rwkv")).resolve()
        distribution_root = package_root.parent

    if module_path.parent != package_root:
        raise FlashRWKVProvenanceError(
            "imported flash_rwkv module is not owned by the validated distribution"
        )
    if not native_path.is_relative_to(package_root):
        raise FlashRWKVProvenanceError(
            "imported flash_rwkv._C is not owned by the validated distribution"
        )
    return FlashRWKVProvenance(
        repository=repository,
        revision=revision,
        editable=editable,
        distribution_root=distribution_root,
        module_path=module_path,
        native_extension_path=native_path,
    )


_FLASH_RWKV_PREFLIGHT_UNSET = object()
_flash_rwkv_preflight_result: FlashRWKVProvenance | FlashRWKVProvenanceError | object = (
    _FLASH_RWKV_PREFLIGHT_UNSET
)
_flash_rwkv_preflight_lock = Lock()


def _resolve_flash_rwkv_preflight(
    result: FlashRWKVProvenance | FlashRWKVProvenanceError | object,
) -> FlashRWKVProvenance:
    if isinstance(result, FlashRWKVProvenanceError):
        raise result
    if result is _FLASH_RWKV_PREFLIGHT_UNSET:
        raise RuntimeError("FlashRWKV preflight result is uninitialized")
    return result


def preflight_flash_rwkv_installation(*, refresh: bool = False) -> FlashRWKVProvenance:
    """Validate FlashRWKV once per process, or revalidate explicitly.

    The cached admission result keeps recurrent dispatch O(1). Call with
    ``refresh=True`` after mutating an editable provider checkout.
    """
    global _flash_rwkv_preflight_result

    result = _flash_rwkv_preflight_result
    if not refresh and result is not _FLASH_RWKV_PREFLIGHT_UNSET:
        return _resolve_flash_rwkv_preflight(result)
    with _flash_rwkv_preflight_lock:
        result = _flash_rwkv_preflight_result
        if refresh or result is _FLASH_RWKV_PREFLIGHT_UNSET:
            try:
                result = validate_flash_rwkv_installation()
            except FlashRWKVProvenanceError as error:
                result = error
            _flash_rwkv_preflight_result = result
        return _resolve_flash_rwkv_preflight(result)


@cache
def _load_flash_rwkv_provider() -> ModuleType:
    """Resolve the admitted provider once for recurrent hot-path reuse."""

    preflight_flash_rwkv_installation()
    return importlib.import_module("flash_rwkv")


class FlashRWKVBackend(BaseBackend):
    """Exact FlashRWKV recurrent backend."""

    backend_type = "flash_rwkv"
    package_name = "flash_rwkv"
    env_var = "FLA_FLASH_RWKV"
    default_enable = True
    priority = 3

    @classmethod
    def is_available(cls) -> bool:
        try:
            preflight_flash_rwkv_installation()
        except FlashRWKVProvenanceError:
            return False
        return True

    def _recurrent_verifier(
        self,
        r: torch.Tensor,
        decay: torch.Tensor,
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
    ) -> tuple[bool, str | None]:
        tensors = (r, decay, k, v, a, b)
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
        if any(tensor.shape != r.shape for tensor in (decay, k, a, b)) or v.shape[:3] != r.shape[:3]:
            return False, "FlashRWKV requires matching [B, T, H, D] input shapes"
        if any(dimension <= 0 for dimension in r.shape) or v.shape[-1] <= 0:
            return False, "FlashRWKV requires positive B, T, H, K, and V dimensions"
        supported_head_sizes = {64, 128, 256}
        if r.shape[-1] != v.shape[-1] or r.shape[-1] not in supported_head_sizes:
            return False, (
                "FlashRWKV requires equal K and V in {64, 128, 256}, "
                f"got K={r.shape[-1]}, V={v.shape[-1]}"
            )
        try:
            if not math.isfinite(float(scale)):
                return False, "FlashRWKV requires a finite scale"
        except (TypeError, ValueError):
            return False, "FlashRWKV requires a finite scale"
        if mode not in {"fp32io16", "fp16"}:
            return False, "FlashRWKV mode must be 'fp32io16' or 'fp16'"
        if mode == "fp16" and any(tensor.dtype != torch.float16 for tensor in tensors):
            return False, "FlashRWKV mode='fp16' requires float16 inputs"
        if decay_bias is not None:
            if (
                decay_bias.ndim not in {1, 2}
                or decay_bias.numel() != r.shape[2] * r.shape[3]
            ):
                return False, "FlashRWKV decay_bias must have shape [H, D] or flattened [H*D]"
            if not decay_bias.is_cuda or decay_bias.device != r.device:
                return False, "FlashRWKV decay_bias must be on the input CUDA device"
            if not decay_bias.is_contiguous() or decay_bias.dtype != r.dtype:
                return False, "FlashRWKV decay_bias must be contiguous and match the token dtype"
        if elapsed_t is not None:
            if mode != "fp16":
                return False, "FlashRWKV elapsed_t is only valid for mode='fp16'"
            if elapsed_t.ndim != 1 or elapsed_t.dtype != torch.int32:
                return False, "FlashRWKV elapsed_t must be a rank-1 int32 tensor"
            if not elapsed_t.is_cuda or elapsed_t.device != r.device or not elapsed_t.is_contiguous():
                return False, "FlashRWKV elapsed_t must be contiguous on the input CUDA device"
        requires_grad = any(tensor.requires_grad for tensor in tensors) or (
            initial_state is not None and initial_state.requires_grad
        )
        if requires_grad and (decay_bias is not None or elapsed_t is not None):
            return False, "FlashRWKV training requires combined decay logits and no elapsed_t"
        if requires_grad and validated_metadata is not None:
            return False, "FlashRWKV training does not accept validated packed metadata"
        if requires_grad and cu_seqlens is not None:
            return False, "FlashRWKV packed execution is forward-only"
        if requires_grad and mode != "fp32io16":
            return False, "FlashRWKV recurrent autograd requires mode='fp32io16'"
        if cu_seqlens_cpu is not None:
            return False, "FlashRWKV recurrent API does not accept duplicate CPU packed metadata"
        expected_state_rows = r.shape[0]
        if cu_seqlens is not None:
            if r.shape[0] != 1:
                return False, "FlashRWKV packed execution requires B=1"
            if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
                return False, "FlashRWKV cu_seqlens must be a rank-1 int32 tensor"
            if not cu_seqlens.is_contiguous():
                return False, "FlashRWKV cu_seqlens must be contiguous"
            if cu_seqlens.device != r.device:
                return False, "FlashRWKV cu_seqlens must be on the input CUDA device"
            if cu_seqlens.shape[0] < 2:
                return False, "FlashRWKV cu_seqlens must describe at least one sequence"
            expected_state_rows = cu_seqlens.shape[0] - 1
            if state_indices is not None:
                if initial_state is None:
                    return False, "FlashRWKV state_indices requires an initial state pool"
                if not output_final_state:
                    return False, "FlashRWKV stateful packed execution requires output_final_state=True"
                if state_indices.ndim != 1 or state_indices.dtype != torch.int32:
                    return False, "FlashRWKV state_indices must be a rank-1 int32 tensor"
                if not state_indices.is_contiguous():
                    return False, "FlashRWKV state_indices must be contiguous"
                if state_indices.device != r.device:
                    return False, "FlashRWKV state_indices must be on the input CUDA device"
                if state_indices.shape[0] != expected_state_rows:
                    return False, "FlashRWKV state_indices length must match the packed sequence count"
        elif state_indices is not None:
            return False, "FlashRWKV state_indices requires cu_seqlens"
        if validated_metadata is not None and (cu_seqlens is None or state_indices is None):
            return False, "FlashRWKV validated_metadata requires packed stateful metadata"
        if initial_state is not None:
            expected_trailing_shape = (r.shape[2], r.shape[3], v.shape[3])
            if (
                initial_state.ndim != 4
                or initial_state.shape[1:] != expected_trailing_shape
                or (state_indices is None and initial_state.shape[0] != expected_state_rows)
                or (state_indices is not None and initial_state.shape[0] < expected_state_rows)
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
                expected_state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
                if initial_state.dtype != expected_state_dtype:
                    return False, f"FlashRWKV stateful {mode} requires a {expected_state_dtype} state pool"
        if cp_context is not None:
            return False, "FlashRWKV recurrent API does not support context parallel execution"
        if safe_gate:
            return False, "FlashRWKV recurrent API does not expose the FLA safe_gate contract"
        if chunk_size is not None:
            return False, "FlashRWKV recurrent API does not accept chunk_size"
        if disable_recompute:
            return False, "FlashRWKV recurrent API does not support disable_recompute=True"
        if kwargs:
            return False, f"FlashRWKV does not support extra arguments: {sorted(kwargs)}"
        return True, None

    def recurrent_rwkv7_verifier(
        self,
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
    ) -> tuple[bool, str | None]:
        return self._recurrent_verifier(
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
            cu_seqlens_cpu=cu_seqlens_cpu,
            state_indices=state_indices,
            mode=mode,
            safe_gate=safe_gate,
            chunk_size=chunk_size,
            disable_recompute=disable_recompute,
            cp_context=cp_context,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
            **kwargs,
        )

    def prepare_recurrent_metadata(
        self,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        total_tokens: int,
        state_pool_size: int,
    ) -> object:
        flash_rwkv = _load_flash_rwkv_provider()
        return flash_rwkv.prepare_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=total_tokens,
            state_pool_size=state_pool_size,
        )

    def recurrent_rwkv7(
        self,
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
        del cu_seqlens_cpu, safe_gate, chunk_size, disable_recompute, cp_context, kwargs
        set_last_rwkv7_provider(None)
        set_last_rwkv7_kernel(None)
        flash_rwkv = _load_flash_rwkv_provider()
        requires_grad = any(
            tensor is not None and tensor.requires_grad
            for tensor in (r, decay_logits, k, v, a, b, initial_state)
        )
        if requires_grad and validated_metadata is not None:
            raise ValueError("validated_metadata is only supported for packed inference")
        if state_indices is not None:
            kernel = "rwkv7_recurrent_stateful"
            output = flash_rwkv.rwkv7_recurrent_stateful(
                r,
                decay_logits,
                k,
                v,
                a,
                b,
                state_pool=initial_state,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                scale=scale,
                mode=mode,
                decay_bias=decay_bias,
                elapsed_t=elapsed_t,
                validated_metadata=validated_metadata,
            )
            output = (output, initial_state)
        elif requires_grad:
            kernel = "pretrain_recurrent_fp32io16_forward"
            output = flash_rwkv.pretrain_recurrent_fp32io16_forward(
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
        else:
            kernel = "rwkv7_recurrent"
            output = flash_rwkv.rwkv7_recurrent(
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
                state_indices=None,
                mode=mode,
                decay_bias=decay_bias,
                elapsed_t=elapsed_t,
                validated_metadata=validated_metadata,
            )
        set_last_rwkv7_provider("flash_rwkv")
        set_last_rwkv7_kernel(kernel)
        return output

__all__ = [
    "FLASH_RWKV_PUBLIC_OPERATORS",
    "FLASH_RWKV_REPOSITORY",
    "FLASH_RWKV_REQUIRED_OPERATORS",
    "FLASH_RWKV_SOURCE_REVISION",
    "FlashRWKVBackend",
    "FlashRWKVProvenance",
    "FlashRWKVProvenanceError",
    "_load_flash_rwkv_provider",
    "preflight_flash_rwkv_installation",
    "validate_flash_rwkv_installation",
]
