# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Optional FlashRWKV backend for RWKV7 chunk execution."""

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
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import torch

from fla.ops.backends import BaseBackend
from fla.ops.rwkv7.backends.provider import set_last_rwkv7_provider

if TYPE_CHECKING:
    from fla.ops.cp import FLACPContext

FLASH_RWKV_SOURCE_REVISION = "c637985558c398de1db6a3c0523b1eec206a88d4"
FLASH_RWKV_REPOSITORY = "https://github.com/rwkv-rs/FlashRWKV.git"
FLASH_RWKV_EVIDENCE_REVISION = "994f4dd7ece2e5113de1dde79d809a3a7cb332a5"
FLASH_RWKV_EVIDENCE_RUN_ID = 30742281594
FLASH_RWKV_EVIDENCE_ARTIFACT_ID = 8831773079
FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST = (
    "sha256:e6b9baf6e6a627896348fd136c7d94b6404604c3b10d2a78b64de347a3be18b5"
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
    candidate = url.removeprefix("git+").rstrip("/")
    if candidate == FLASH_RWKV_REPOSITORY:
        return FLASH_RWKV_REPOSITORY
    if candidate in {
        "git@github.com:rwkv-rs/FlashRWKV.git",
        "ssh://git@github.com/rwkv-rs/FlashRWKV.git",
    }:
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
    try:
        parameters = inspect.signature(module.rwkv7).parameters
        stateful_parameters = inspect.signature(module.rwkv7_recurrent_stateful).parameters
    except (AttributeError, TypeError, ValueError) as error:
        raise FlashRWKVProvenanceError("FlashRWKV public RWKV7 API is unavailable") from error
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
    if not required_parameters <= parameters.keys():
        raise FlashRWKVProvenanceError("FlashRWKV rwkv7 signature is incompatible")
    if not required_stateful_parameters <= stateful_parameters.keys():
        raise FlashRWKVProvenanceError(
            "FlashRWKV rwkv7_recurrent_stateful signature is incompatible"
        )
    if not callable(getattr(module, "validate_packed_metadata_strict", None)):
        raise FlashRWKVProvenanceError(
            "FlashRWKV strict packed metadata validator is unavailable"
        )


def validate_flash_rwkv_installation() -> FlashRWKVProvenance:
    """Validate exact distribution, Python module, and native extension ownership.

    Editable installs are intentionally revalidated on every call so a dirty checkout,
    moved HEAD, or changed origin cannot remain hidden behind a process-wide cache.
    """

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
        if repository is None or source != FLASH_RWKV_REPOSITORY:
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


class FlashRWKVBackend(BaseBackend):
    """FlashRWKV FP32-state chunk backend."""

    backend_type = "flash_rwkv"
    package_name = "flash_rwkv"
    env_var = "FLA_FLASH_RWKV"
    default_enable = True
    fail_closed_on_explicit_enable = True
    priority = 3

    def on_explicit_failure(self) -> None:
        set_last_rwkv7_provider(None)

    @classmethod
    def is_available(cls) -> bool:
        try:
            validate_flash_rwkv_installation()
        except FlashRWKVProvenanceError:
            return False
        return True

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
        if cu_seqlens_cpu is not None:
            return False, "FlashRWKV does not accept duplicate CPU packed metadata"
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
            output = (output, initial_state)
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


__all__ = [
    "FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST",
    "FLASH_RWKV_EVIDENCE_ARTIFACT_ID",
    "FLASH_RWKV_EVIDENCE_REVISION",
    "FLASH_RWKV_EVIDENCE_RUN_ID",
    "FLASH_RWKV_REPOSITORY",
    "FLASH_RWKV_SOURCE_REVISION",
    "FlashRWKVBackend",
    "FlashRWKVProvenance",
    "FlashRWKVProvenanceError",
    "validate_flash_rwkv_installation",
]
