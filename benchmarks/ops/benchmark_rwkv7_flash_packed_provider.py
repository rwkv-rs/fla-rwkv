# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness-gated packed serving benchmark through public ``recurrent_rwkv7``."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType

import torch

from fla.ops.rwkv7 import get_last_rwkv7_provider, recurrent_rwkv7
from fla.ops.rwkv7.backends.flash_rwkv import (
    FLASH_RWKV_SOURCE_REVISION,
    validate_flash_rwkv_installation,
)

FLASH_RWKV_EVIDENCE_REVISION = "71dd68897ffa79b727409b299ddea2c0eaba2563"
FLASH_RWKV_EVIDENCE_RUN_ID = 30751092211
FLASH_RWKV_EVIDENCE_ARTIFACT_ID = 8834636910
FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST = (
    "sha256:1640178254d96db98c60adf630372d60be7e797a21896a33613da69cc8823542"
)
HEAD_SIZE = 64
PROFILES: dict[str, tuple[int, ...]] = {
    "decode_b320": (1,) * 320,
    "equal16_b320": (16,) * 320,
    "ragged16_b320": tuple(range(1, 17)) * 20,
}
RRMSE_LIMITS = {
    "fp32io16": {"output": 0.002, "state": 0.002},
    "fp16": {"output": 0.003, "state": 0.003},
}
RESULT_FIELDS = (
    "label",
    "B",
    "T",
    "iters",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "tok_s_p50",
)
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Payload:
    sequence_lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    state_slots: tuple[int, ...]
    inputs: tuple[torch.Tensor, ...]
    cu_seqlens: torch.Tensor
    state_indices: torch.Tensor
    initial_state_pool: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return self.offsets[-1]


def _repository_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _hardware(runner_label: str) -> dict[str, object]:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "runner_label": runner_label,
        "device_index": index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((error / baseline).item())


def _make_payload(
    sequence_lengths: tuple[int, ...],
    *,
    hidden_size: int,
    mode: str,
    seed: int,
) -> Payload:
    heads = hidden_size // HEAD_SIZE
    total_tokens = sum(sequence_lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(scale: float) -> torch.Tensor:
        return (
            scale
            * torch.randn(
                (1, total_tokens, heads, HEAD_SIZE),
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
        ).to(torch.float16)

    inputs = tuple(normal(0.02) for _ in range(6))
    inputs = (
        inputs[0],
        (
            -0.05
            - 0.15
            * torch.rand(
                inputs[1].shape,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
        ).to(torch.float16),
        *inputs[2:],
    )
    offsets = [0]
    for length in sequence_lengths:
        offsets.append(offsets[-1] + length)
    state_slots = tuple(range(len(sequence_lengths) - 1, -1, -1))
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    state_indices = torch.tensor(state_slots, device="cuda", dtype=torch.int32)
    state_dtype = torch.float32 if mode == "fp32io16" else torch.float16
    initial_state_pool = (
        0.02
        * torch.randn(
            (len(sequence_lengths) + 7, heads, HEAD_SIZE, HEAD_SIZE),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).to(state_dtype)
    return Payload(
        sequence_lengths=sequence_lengths,
        offsets=tuple(offsets),
        state_slots=state_slots,
        inputs=inputs,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        initial_state_pool=initial_state_pool,
    )


def _oracle(payload: Payload) -> tuple[torch.Tensor, torch.Tensor]:
    r, w, k, v, a, b = (tensor[0] for tensor in payload.inputs)
    state_pool = payload.initial_state_pool.float().clone()
    output = torch.empty_like(v, dtype=torch.float32)
    max_length = max(payload.sequence_lengths)
    for token_offset in range(max_length):
        active_sequences = tuple(
            index
            for index, length in enumerate(payload.sequence_lengths)
            if token_offset < length
        )
        token_indices = torch.tensor(
            [payload.offsets[index] + token_offset for index in active_sequences],
            device="cuda",
            dtype=torch.long,
        )
        slots = torch.tensor(
            [payload.state_slots[index] for index in active_sequences],
            device="cuda",
            dtype=torch.long,
        )
        previous_state = state_pool.index_select(0, slots)
        token_r = r.index_select(0, token_indices).float()
        token_w = w.index_select(0, token_indices).float()
        token_k = k.index_select(0, token_indices).float()
        token_v = v.index_select(0, token_indices).float()
        token_a = a.index_select(0, token_indices).float()
        token_b = b.index_select(0, token_indices).float()
        a_state = torch.einsum("nhk,nhkv->nhv", token_a, previous_state)
        updated_state = (
            token_w.exp().unsqueeze(-1) * previous_state
            + token_b.unsqueeze(-1) * a_state.unsqueeze(-2)
            + token_k.unsqueeze(-1) * token_v.unsqueeze(-2)
        )
        output.index_copy_(
            0,
            token_indices,
            torch.einsum("nhk,nhkv->nhv", token_r, updated_state),
        )
        state_pool.index_copy_(0, slots, updated_state)
    return output.to(v.dtype).unsqueeze(0), state_pool


def _launch(
    payload: Payload,
    state_pool: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = recurrent_rwkv7(
        *payload.inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=payload.cu_seqlens,
        state_indices=payload.state_indices,
        mode=mode,
    )
    if get_last_rwkv7_provider() != "flash_rwkv":
        raise RuntimeError("public packed call did not select FlashRWKV")
    if final_state is not state_pool:
        raise RuntimeError("public packed call did not return its input state pool")
    return output, final_state


def _cuda_graph_evidence(
    payload: Payload,
    *,
    mode: str,
    observed: dict[str, torch.Tensor],
) -> dict[str, object]:
    state_pool = payload.initial_state_pool.clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            _launch(payload, state_pool, mode=mode)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_state = _launch(payload, state_pool, mode=mode)
    graph.replay()
    torch.cuda.synchronize()
    return {
        "cuda_graph_capture_succeeded": True,
        "captured_output_finite": bool(torch.isfinite(captured_output).all().item()),
        "final_state_identity_preserved": captured_state is state_pool,
        "cu_seqlens_identity_preserved": observed.get("cu_seqlens") is payload.cu_seqlens,
        "state_indices_identity_preserved": observed.get("state_indices") is payload.state_indices,
        "cu_seqlens_data_ptr": payload.cu_seqlens.data_ptr(),
        "state_indices_data_ptr": payload.state_indices.data_ptr(),
        "interpretation": (
            "CUDA Graph capture through fla.ops.rwkv7.recurrent_rwkv7 proves the "
            "packed launch boundary performs no device-to-host synchronization"
        ),
    }


def _run_case(
    profile: str,
    *,
    provider_module: ModuleType,
    mode: str,
    hidden_size: int,
    warmup: int,
    iters: int,
    seed: int,
    observed: dict[str, torch.Tensor],
) -> dict[str, object]:
    payload = _make_payload(
        PROFILES[profile],
        hidden_size=hidden_size,
        mode=mode,
        seed=seed,
    )
    provider_module.validate_packed_metadata_strict(
        payload.cu_seqlens,
        payload.state_indices,
        total_tokens=payload.total_tokens,
        state_pool_size=payload.initial_state_pool.shape[0],
    )
    expected_output, expected_pool = _oracle(payload)
    first_pool = payload.initial_state_pool.clone()
    second_pool = payload.initial_state_pool.clone()
    untouched = torch.arange(
        len(payload.sequence_lengths),
        payload.initial_state_pool.shape[0],
        device="cuda",
        dtype=torch.long,
    )
    first_output, first_final_state = _launch(payload, first_pool, mode=mode)
    first_call_identity_preserved = bool(
        observed.get("state_pool") is first_pool
        and observed.get("cu_seqlens") is payload.cu_seqlens
        and observed.get("state_indices") is payload.state_indices
    )
    second_output, _ = _launch(payload, second_pool, mode=mode)
    torch.cuda.synchronize()
    active_slots = payload.state_indices.long()
    output_error = _relative_rmse(first_output, expected_output)
    state_error = _relative_rmse(
        first_pool.index_select(0, active_slots),
        expected_pool.index_select(0, active_slots),
    )
    finite = bool(
        torch.isfinite(first_output).all().item()
        and torch.isfinite(first_pool.index_select(0, active_slots)).all().item()
    )
    deterministic = bool(
        torch.equal(first_output, second_output)
        and torch.equal(first_pool, second_pool)
    )
    untouched_preserved = bool(
        torch.equal(
            first_pool.index_select(0, untouched),
            payload.initial_state_pool.index_select(0, untouched),
        )
    )
    selected_rows_updated = not torch.equal(
        first_pool.index_select(0, active_slots),
        payload.initial_state_pool.index_select(0, active_slots),
    )
    limits = RRMSE_LIMITS[mode]
    correct = bool(
        finite
        and deterministic
        and untouched_preserved
        and selected_rows_updated
        and first_final_state is first_pool
        and first_call_identity_preserved
        and output_error <= limits["output"]
        and state_error <= limits["state"]
    )
    if not correct:
        raise RuntimeError(
            f"packed correctness gate failed: profile={profile} mode={mode} "
            f"output_rrmse={output_error} state_rrmse={state_error}"
        )

    measurement_pool = payload.initial_state_pool.clone()
    for _ in range(warmup):
        measurement_pool.copy_(payload.initial_state_pool)
        _launch(payload, measurement_pool, mode=mode)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        measurement_pool.copy_(payload.initial_state_pool)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        _launch(payload, measurement_pool, mode=mode)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    p10 = _percentile(samples, 0.1)
    p50 = statistics.median(samples)
    p90 = _percentile(samples, 0.9)
    graph_evidence = (
        _cuda_graph_evidence(payload, mode=mode, observed=observed)
        if profile == "decode_b320"
        else None
    )
    return {
        "label": f"packed-{profile}-{mode}",
        "profile": profile,
        "mode": mode,
        "provider": "flash_rwkv",
        "public_api": "fla.ops.rwkv7.recurrent_rwkv7",
        "oracle": "explicit-pytorch-recurrence",
        "B": len(payload.sequence_lengths),
        "T": payload.total_tokens,
        "H": hidden_size // HEAD_SIZE,
        "head_size": HEAD_SIZE,
        "state_pool_rows": payload.initial_state_pool.shape[0],
        "state_slot_mapping": "reverse sequence order with seven untouched rows",
        "state_dtype": str(payload.initial_state_pool.dtype),
        "token_dtype": str(payload.inputs[0].dtype),
        "warmup": warmup,
        "iters": iters,
        "p10_ms": p10,
        "p50_ms": p50,
        "p90_ms": p90,
        "tok_s_p50": payload.total_tokens * 1000.0 / p50,
        "raw_samples_ms": samples,
        "correctness": {
            "passed": correct,
            "finite": finite,
            "deterministic": deterministic,
            "output_relative_rmse": output_error,
            "output_relative_rmse_limit": limits["output"],
            "final_state_relative_rmse": state_error,
            "final_state_relative_rmse_limit": limits["state"],
            "final_state_is_state_pool": first_final_state is first_pool,
            "selected_rows_updated": selected_rows_updated,
            "untouched_slots_preserved": untouched_preserved,
        },
        "device_metadata": {
            "device": str(payload.cu_seqlens.device),
            "cu_seqlens_dtype": str(payload.cu_seqlens.dtype),
            "state_indices_dtype": str(payload.state_indices.dtype),
            "cu_seqlens_identity_preserved": observed.get("cu_seqlens") is payload.cu_seqlens,
            "state_indices_identity_preserved": observed.get("state_indices") is payload.state_indices,
            "cuda_graph_evidence": graph_evidence,
        },
        "measurement": {
            "included": (
                "public recurrent_rwkv7 dispatch, provenance identity check, FlashRWKV "
                "packed provider launch, and device synchronization"
            ),
            "excluded": (
                "input allocation, strict debug metadata validation, oracle, "
                "state reset, warmup, and serialization"
            ),
        },
    }


def _format_result(row: dict[str, object]) -> str:
    missing = tuple(field for field in RESULT_FIELDS if field not in row)
    if missing:
        raise ValueError(f"packed benchmark row lacks RESULT fields: {missing}")
    return (
        f"RESULT B={row['B']} T={row['T']} iters={row['iters']} "
        f"p10_ms={float(row['p10_ms']):.6f} "
        f"p50_ms={float(row['p50_ms']):.6f} "
        f"p90_ms={float(row['p90_ms']):.6f} "
        f"tok_s_p50={float(row['tok_s_p50']):.6f} "
        f"label={row['label']} provider={row['provider']} mode={row['mode']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp32io16", "fp16"), required=True)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-provider-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.hidden_size <= 0 or args.hidden_size % HEAD_SIZE:
        parser.error("hidden-size must be a positive multiple of 64")
    if args.warmup <= 0 or args.iters <= 0:
        parser.error("warmup and iters must be positive")
    provider_module = importlib.import_module("flash_rwkv")
    source_revision = _repository_revision()
    if source_revision != args.expected_source_revision:
        raise RuntimeError(
            f"source revision mismatch: expected={args.expected_source_revision} actual={source_revision}"
        )
    provenance = validate_flash_rwkv_installation()
    if provenance.revision != args.expected_provider_revision:
        raise RuntimeError(
            f"provider revision mismatch: expected={args.expected_provider_revision} actual={provenance.revision}"
        )

    observed: dict[str, torch.Tensor] = {}
    original = provider_module.rwkv7_recurrent_stateful

    @wraps(original)
    def observe_metadata(*call_args, **call_kwargs):
        observed["state_pool"] = call_kwargs["state_pool"]
        observed["cu_seqlens"] = call_kwargs["cu_seqlens"]
        observed["state_indices"] = call_kwargs["state_indices"]
        return original(*call_args, **call_kwargs)

    provider_module.rwkv7_recurrent_stateful = observe_metadata
    try:
        results = [
            _run_case(
                profile,
                provider_module=provider_module,
                mode=args.mode,
                hidden_size=args.hidden_size,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + index * 1009,
                observed=observed,
            )
            for index, profile in enumerate(args.profiles)
        ]
    finally:
        provider_module.rwkv7_recurrent_stateful = original
    payload = {
        "schema_version": 1,
        "benchmark": "fla_rwkv7_flash_packed_serving",
        "pr_number": args.pr_number,
        "source_revision": source_revision,
        "flash_rwkv_source_revision": provenance.revision,
        "flash_rwkv_repository": provenance.repository,
        "flash_rwkv_native_extension": str(provenance.native_extension_path),
        "flash_rwkv_clean_evidence": {
            "evidence_revision": FLASH_RWKV_EVIDENCE_REVISION,
            "workflow_run_id": FLASH_RWKV_EVIDENCE_RUN_ID,
            "artifact_id": FLASH_RWKV_EVIDENCE_ARTIFACT_ID,
            "artifact_digest": FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST,
            "runtime_semantic_revision": FLASH_RWKV_SOURCE_REVISION,
        },
        "runner_label": args.runner_label,
        "hardware": _hardware(args.runner_label),
        "mode": args.mode,
        "all_cases_correct": all(row["correctness"]["passed"] for row in results),
        "result_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in results:
        print(_format_result(row))
    print(json.dumps({"output": str(args.output), "all_cases_correct": payload["all_cases_correct"]}, sort_keys=True))


if __name__ == "__main__":
    main()
