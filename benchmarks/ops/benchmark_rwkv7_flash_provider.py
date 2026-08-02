# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Correctness and diagnostic benchmark contract for the FlashRWKV adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import torch

from fla.ops.rwkv7 import chunk_rwkv7, get_last_rwkv7_provider
from fla.utils import device


def _source_provenance() -> tuple[str, str]:
    distribution = importlib.metadata.distribution("flash-rwkv")
    direct_url = distribution.read_text("direct_url.json")
    metadata = json.loads(direct_url) if direct_url else {}
    source = metadata.get("url", "unknown")
    revision = metadata.get("vcs_info", {}).get("commit_id", "unknown")
    if revision != "unknown":
        return source, revision

    revision = "unknown"
    if source.startswith("file://"):
        path = Path(source.removeprefix("file://"))
        with suppress(OSError, subprocess.CalledProcessError):
            revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return source, revision


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


def _run(provider: str, inputs: list[torch.Tensor], state: torch.Tensor):
    if provider == "flash_rwkv":
        os.environ["FLA_FLASH_RWKV"] = "1"
    else:
        os.environ.pop("FLA_FLASH_RWKV", None)
    output, final_state = chunk_rwkv7(*inputs, initial_state=state, output_final_state=True, chunk_size=16)
    return output, final_state, get_last_rwkv7_provider()


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
    state = torch.zeros(args.batch_size, args.heads, 64, 64, device=device, dtype=torch.float32)

    expected, expected_state, baseline_provider = _run("fla", inputs, state)
    actual, actual_state, selected_provider = _run("flash_rwkv", inputs, state)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=2e-3)
    for _ in range(args.warmup):
        _run("flash_rwkv", inputs, state)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.iters):
        start = time.perf_counter_ns()
        _run("flash_rwkv", inputs, state)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    quantiles = torch.tensor(samples).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
    provider_source, provider_revision = _source_provenance()
    source_revision = _repository_revision()
    if args.expected_source_revision and source_revision != args.expected_source_revision:
        raise RuntimeError(f"source revision mismatch: expected={args.expected_source_revision} actual={source_revision}")
    if args.expected_provider_revision and provider_revision != args.expected_provider_revision:
        raise RuntimeError(
            f"FlashRWKV revision mismatch: expected={args.expected_provider_revision} actual={provider_revision}"
        )
    report = {
        "schema_version": 1,
        "pr_number": args.pr_number,
        "source_revision": source_revision,
        "backend": selected_provider,
        "reference_backend": baseline_provider,
        "selected_provider": selected_provider,
        "baseline_provider": baseline_provider,
        "flash_rwkv_version": importlib.metadata.version("flash-rwkv"),
        "flash_rwkv_source": provider_source,
        "flash_rwkv_source_revision": provider_revision,
        "hardware": _hardware(args.runner_label),
        "measurement": {
            "included": "chunk_rwkv7 adapter dispatch, FlashRWKV provider execution, and device synchronization",
            "excluded": "input allocation, correctness baseline, warmup, provenance collection, and report serialization",
        },
        "dtype": args.dtype,
        "head_size": 64,
        "B": args.batch_size,
        "T": args.tokens,
        "H": args.heads,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms": {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]},
        "tokens_per_second": args.batch_size * args.tokens / (quantiles[1] / 1000),
        "output_error": _error_summary(actual, expected),
        "final_state_error": _error_summary(actual_state, expected_state),
    }
    if selected_provider != "flash_rwkv" or baseline_provider != "fla":
        raise RuntimeError(f"provider selection contract failed: {report}")
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
