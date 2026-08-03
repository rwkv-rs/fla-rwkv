# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness and diagnostic benchmark contract for the FlashRWKV adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from fla.ops.rwkv7 import get_last_rwkv7_kernel, get_last_rwkv7_provider, recurrent_rwkv7
from fla.ops.rwkv7.backends.flash_rwkv import validate_flash_rwkv_installation
from fla.utils import device

RESULT_FIELDS = ("label", "B", "T", "iters", "p10_ms", "p50_ms", "p90_ms", "tok_s_p50")


def _format_result(row: dict[str, object]) -> str:
    missing = tuple(field for field in RESULT_FIELDS if field not in row)
    if missing:
        raise ValueError(f"benchmark row is missing RESULT fields: {missing}")

    def metric(field: str) -> str:
        return str(round(float(row[field]), 6))

    return (
        f"RESULT B={row['B']} T={row['T']} iters={row['iters']} "
        f"p10_ms={metric('p10_ms')} p50_ms={metric('p50_ms')} "
        f"p90_ms={metric('p90_ms')} tok_s_p50={metric('tok_s_p50')} "
        f"label={row['label']}"
    )


def _source_provenance() -> tuple[str, str, str]:
    provenance = validate_flash_rwkv_installation()
    return (
        provenance.repository,
        provenance.revision,
        str(provenance.native_extension_path),
    )


def _repository_revision() -> str:
    repository = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _hardware(runner_label: str) -> dict[str, str | int | list[int]]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "runner_label": runner_label,
        "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
    }


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-12)
    return {
        "max_abs": error.max().item(),
        "mean_abs": error.mean().item(),
        "max_rel": (error / denominator).max().item(),
    }


def _run_provider(inputs: list[torch.Tensor], state: torch.Tensor):
    os.environ.pop("FLA_FLASH_RWKV", None)
    output, final_state = recurrent_rwkv7(
        *inputs,
        initial_state=state,
        output_final_state=True,
    )
    return output, final_state, get_last_rwkv7_provider(), get_last_rwkv7_kernel()


def _run_oracle(
    inputs: list[torch.Tensor],
    state: torch.Tensor,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the RWKV7 cell sequentially with ordinary PyTorch ops."""
    r, decay_logits, k, v, a, b = inputs
    log_decay = -0.6065306597126334 * decay_logits.float().sigmoid()
    recurrent_state = state.float()
    outputs = []
    for token_index in range(r.shape[1]):
        previous_state = recurrent_state
        token_a = a[:, token_index].float()
        a_state = torch.einsum("bhk,bhkv->bhv", token_a, previous_state)
        recurrent_state = (
            log_decay[:, token_index].float().exp().unsqueeze(-1) * previous_state
            + b[:, token_index].float().unsqueeze(-1) * a_state.unsqueeze(-2)
            + k[:, token_index].float().unsqueeze(-1) * v[:, token_index].float().unsqueeze(-2)
        )
        outputs.append(
            scale
            * torch.einsum(
                "bhk,bhkv->bhv",
                r[:, token_index].float(),
                recurrent_state,
            )
        )
    return torch.stack(outputs, dim=1).to(v.dtype), recurrent_state


def _training_inputs(
    inputs: list[torch.Tensor],
    state: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    return (
        [tensor.detach().clone().requires_grad_() for tensor in inputs],
        state.detach().clone().requires_grad_(),
    )


def _backward(output: torch.Tensor, final_state: torch.Tensor) -> None:
    (output.float().square().mean() + final_state.float().square().mean()).backward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner-label", default=os.environ.get("RWKV7_BENCH_RUNNER_LABEL", "local"))
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-provider-revision")
    args = parser.parse_args()
    if min(args.batch_size, args.tokens, args.heads, args.warmup, args.iters) <= 0:
        parser.error("shape, warmup, and iteration values must be positive")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(7)
    shape = (args.batch_size, args.tokens, args.heads, 64)
    inputs = [(torch.randn(shape, device=device, dtype=dtype) * 0.02).contiguous() for _ in range(6)]
    inputs[1] = torch.full_like(inputs[1], -0.1)
    state = torch.randn(
        args.batch_size,
        args.heads,
        64,
        64,
        device=device,
        dtype=torch.float32,
    ) * 0.01

    oracle_inputs, oracle_state = _training_inputs(inputs, state)
    provider_inputs, provider_state = _training_inputs(inputs, state)
    expected, expected_state = _run_oracle(oracle_inputs, oracle_state)
    _backward(expected, expected_state)
    actual, actual_state, selected_provider, selected_kernel = _run_provider(
        provider_inputs,
        provider_state,
    )
    _backward(actual, actual_state)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=2e-3)
    for actual_input, expected_input in zip(provider_inputs, oracle_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=5e-2, atol=5e-3)
    torch.testing.assert_close(provider_state.grad, oracle_state.grad, rtol=5e-2, atol=5e-3)
    for _ in range(args.warmup):
        warmup_inputs, warmup_state = _training_inputs(inputs, state)
        warmup_output, warmup_final_state, _, _ = _run_provider(warmup_inputs, warmup_state)
        _backward(warmup_output, warmup_final_state)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        sample_inputs, sample_state = _training_inputs(inputs, state)
        start = time.perf_counter_ns()
        sample_output, sample_final_state, _, _ = _run_provider(sample_inputs, sample_state)
        _backward(sample_output, sample_final_state)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    quantiles = torch.tensor(samples).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
    provider_source, provider_revision, native_extension_path = _source_provenance()
    source_revision = _repository_revision()
    if args.expected_source_revision and source_revision != args.expected_source_revision:
        raise RuntimeError(f"source revision mismatch: expected={args.expected_source_revision} actual={source_revision}")
    if args.expected_provider_revision and provider_revision != args.expected_provider_revision:
        raise RuntimeError(
            f"FlashRWKV revision mismatch: expected={args.expected_provider_revision} actual={provider_revision}"
        )
    report = {
        "schema_version": 1,
        "label": f"flash-rwkv-raw-decay-recurrent-autograd-{args.dtype}-B{args.batch_size}T{args.tokens}",
        "pr_number": args.pr_number,
        "source_revision": source_revision,
        "backend": selected_provider,
        "selected_provider": selected_provider,
        "selected_kernel": selected_kernel,
        "oracle": "independent-raw-decay-transform-plus-pytorch-sequential-recurrence-autograd",
        "flash_rwkv_source": provider_source,
        "flash_rwkv_source_revision": provider_revision,
        "flash_rwkv_native_extension": native_extension_path,
        "hardware": _hardware(args.runner_label),
        "measurement": {
            "included": (
                "raw-decay recurrent_rwkv7 dispatch, fused FlashRWKV decay transform plus "
                "recurrent forward and backward, and device synchronization"
            ),
            "excluded": "input cloning, correctness oracle, warmup, provenance collection, and report serialization",
        },
        "dtype": args.dtype,
        "head_size": 64,
        "B": args.batch_size,
        "T": args.tokens,
        "H": args.heads,
        "warmup": args.warmup,
        "iters": args.iters,
        "p10_ms": quantiles[0],
        "p50_ms": quantiles[1],
        "p90_ms": quantiles[2],
        "tok_s_p50": args.batch_size * args.tokens / (quantiles[1] / 1000),
        "latency_ms": {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]},
        "tokens_per_second": args.batch_size * args.tokens / (quantiles[1] / 1000),
        "output_error": _error_summary(actual, expected),
        "final_state_error": _error_summary(actual_state, expected_state),
        "input_gradient_error": [
            _error_summary(actual_input.grad, expected_input.grad)
            for actual_input, expected_input in zip(provider_inputs, oracle_inputs, strict=True)
        ],
        "initial_state_gradient_error": _error_summary(provider_state.grad, oracle_state.grad),
    }
    if selected_provider != "flash_rwkv":
        raise RuntimeError(f"provider selection contract failed: {report}")
    if selected_kernel != "pretrain_recurrent_fp32io16_forward":
        raise RuntimeError(f"kernel selection contract failed: {report}")
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(_format_result(report))
    print(serialized)


if __name__ == "__main__":
    main()
