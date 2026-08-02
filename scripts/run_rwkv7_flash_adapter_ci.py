# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Run the CPU gate or the controlled FlashRWKV adapter GPU contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRO6000_RUNNER_LABEL = "rwkv-sha-pro6000x8"
FLASH_RWKV_SOURCE_REVISION = "c637985558c398de1db6a3c0523b1eec206a88d4"
FLASH_RWKV_EVIDENCE_REVISION = "994f4dd7ece2e5113de1dde79d809a3a7cb332a5"
FLASH_RWKV_EVIDENCE_RUN_ID = 30742281594
FLASH_RWKV_EVIDENCE_ARTIFACT_ID = 8831773079
FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST = "sha256:e6b9baf6e6a627896348fd136c7d94b6404604c3b10d2a78b64de347a3be18b5"
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RESULT_FIELDS = ("label", "B", "T", "iters", "p10_ms", "p50_ms", "p90_ms", "tok_s_p50")


def _revision() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _validate_revision(name: str, revision: str) -> None:
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"{name} must be a full lowercase Git revision, got {revision!r}")


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def _quick_gate(source_revision: str) -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with tempfile.TemporaryDirectory(prefix="fla-rwkv7-bytecode-") as bytecode_dir:
        environment["PYTHONPYCACHEPREFIX"] = bytecode_dir
        _run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "fla/ops/rwkv7/backends",
                "fla/ops/rwkv7/chunk.py",
                "fla/ops/rwkv7/recurrent.py",
                "fla/layers/rwkv7.py",
                "fla/models/rwkv7/configuration_rwkv7.py",
                "benchmarks/ops/benchmark_rwkv7_flash_provider.py",
                "benchmarks/ops/benchmark_rwkv7_flash_packed_provider.py",
                "scripts/build_packages.py",
                "scripts/run_rwkv7_flash_adapter_ci.py",
                "tests/racecheck/test_rwkv7_flash_adapter.py",
                "tests/context_parallel/test_cp_rwkv7.py",
                "tests/test_rwkv7_flash_packaging.py",
            ],
            environment=environment,
        )
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--confcutdir=tests/ops",
            "tests/ops/test_rwkv7_backends.py",
            "-k",
            "not real_provider",
        ],
        environment=environment,
    )
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_rwkv7_flash_packaging.py",
        ],
        environment=environment,
    )
    _run(
        [sys.executable, "benchmarks/ops/benchmark_rwkv7_flash_provider.py", "--help"],
        environment=environment,
    )
    _run(
        [sys.executable, "benchmarks/ops/benchmark_rwkv7_flash_packed_provider.py", "--help"],
        environment=environment,
    )
    print(json.dumps({"gate": "quick-cpu-static", "source_revision": source_revision, "passed": True}, sort_keys=True))


def _validate_benchmark(
    report: dict,
    *,
    source_revision: str,
    provider_revision: str,
    runner_label: str,
    pr_number: int,
    dtype: str,
    batch_size: int,
    tokens: int,
    warmup: int,
    iters: int,
) -> None:
    expected = {
        "source_revision": source_revision,
        "pr_number": pr_number,
        "flash_rwkv_source_revision": provider_revision,
        "backend": "flash_rwkv",
        "selected_provider": "flash_rwkv",
        "oracle": "explicit-pytorch-sequential-recurrent-autograd",
        "dtype": dtype,
        "B": batch_size,
        "T": tokens,
        "warmup": warmup,
        "iters": iters,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"benchmark field mismatch for {key}: expected={value!r} actual={report.get(key)!r}")
    if report.get("hardware", {}).get("runner_label") != runner_label:
        raise RuntimeError("benchmark artifact is not bound to the requested runner label")
    if "PRO 6000" not in report["hardware"].get("device_name", ""):
        raise RuntimeError("benchmark artifact was not produced on an NVIDIA RTX PRO 6000")
    if set(report.get("latency_ms", {})) != {"p10", "p50", "p90"}:
        raise RuntimeError("benchmark artifact lacks the required latency percentiles")
    if report.get("tokens_per_second", 0) <= 0:
        raise RuntimeError("benchmark artifact has invalid throughput")
    missing_result_fields = tuple(field for field in RESULT_FIELDS if field not in report)
    if missing_result_fields:
        raise RuntimeError(f"benchmark artifact lacks RESULT fields: {missing_result_fields}")
    for field in ("p10_ms", "p50_ms", "p90_ms", "tok_s_p50"):
        value = report[field]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"benchmark artifact has invalid RESULT field {field}")
    if (
        report["p10_ms"] != report["latency_ms"]["p10"]
        or report["p50_ms"] != report["latency_ms"]["p50"]
        or report["p90_ms"] != report["latency_ms"]["p90"]
        or report["tok_s_p50"] != report["tokens_per_second"]
    ):
        raise RuntimeError("benchmark artifact RESULT fields disagree with compatibility fields")
    for error_name in ("output_error", "final_state_error"):
        if set(report.get(error_name, {})) != {"max_abs", "mean_abs", "max_rel"}:
            raise RuntimeError(f"benchmark artifact lacks {error_name}")
    if len(report.get("input_gradient_error", [])) != 6:
        raise RuntimeError("benchmark artifact lacks six input gradient error summaries")
    if set(report.get("initial_state_gradient_error", {})) != {"max_abs", "mean_abs", "max_rel"}:
        raise RuntimeError("benchmark artifact lacks initial-state gradient error")


def _validate_packed_benchmark(
    report: dict,
    *,
    source_revision: str,
    provider_revision: str,
    runner_label: str,
    pr_number: int,
    mode: str,
) -> None:
    expected = {
        "benchmark": "fla_rwkv7_flash_packed_serving",
        "source_revision": source_revision,
        "pr_number": pr_number,
        "flash_rwkv_source_revision": provider_revision,
        "runner_label": runner_label,
        "mode": mode,
        "all_cases_correct": True,
        "result_count": 3,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"packed benchmark field mismatch for {key}: expected={value!r} actual={report.get(key)!r}"
            )
    evidence = report.get("flash_rwkv_clean_evidence", {})
    expected_evidence = {
        "runtime_semantic_revision": provider_revision,
        "evidence_revision": FLASH_RWKV_EVIDENCE_REVISION,
        "workflow_run_id": FLASH_RWKV_EVIDENCE_RUN_ID,
        "artifact_id": FLASH_RWKV_EVIDENCE_ARTIFACT_ID,
        "artifact_digest": FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST,
    }
    if evidence != expected_evidence:
        raise RuntimeError("packed benchmark does not bind the accepted FlashRWKV evidence")
    hardware = report.get("hardware", {})
    if hardware.get("runner_label") != runner_label or "PRO 6000" not in hardware.get("device_name", ""):
        raise RuntimeError("packed benchmark was not produced on the controlled PRO 6000 runner")
    results = report.get("results", [])
    if {row.get("profile") for row in results} != {
        "decode_b320",
        "equal16_b320",
        "ragged16_b320",
    }:
        raise RuntimeError("packed benchmark profile set is incomplete")
    for row in results:
        if row.get("provider") != "flash_rwkv" or row.get("public_api") != "fla.ops.rwkv7.recurrent_rwkv7":
            raise RuntimeError("packed benchmark did not exercise the public recurrent provider")
        if row.get("oracle") != "explicit-pytorch-recurrence":
            raise RuntimeError("packed benchmark oracle identity is invalid")
        missing = tuple(field for field in RESULT_FIELDS if field not in row)
        if missing:
            raise RuntimeError(f"packed benchmark row lacks RESULT fields: {missing}")
        correctness = row.get("correctness", {})
        for field in (
            "passed",
            "finite",
            "deterministic",
            "final_state_is_state_pool",
            "selected_rows_updated",
            "untouched_slots_preserved",
        ):
            if correctness.get(field) is not True:
                raise RuntimeError(f"packed correctness field {field} is not true")
        metadata = row.get("device_metadata", {})
        if metadata.get("cu_seqlens_identity_preserved") is not True:
            raise RuntimeError("packed cu_seqlens identity was not preserved")
        if metadata.get("state_indices_identity_preserved") is not True:
            raise RuntimeError("packed state_indices identity was not preserved")
        graph = metadata.get("cuda_graph_evidence")
        if row.get("profile") == "decode_b320":
            if not isinstance(graph, dict) or not all(
                graph.get(field) is True
                for field in (
                    "cuda_graph_capture_succeeded",
                    "captured_output_finite",
                    "final_state_identity_preserved",
                    "cu_seqlens_identity_preserved",
                    "state_indices_identity_preserved",
                )
            ):
                raise RuntimeError("packed decode lacks no-host-sync CUDA Graph evidence")


def _validate_racecheck(path: Path, provider_revision: str) -> dict[str, object]:
    log = path.read_text(encoding="utf-8")
    if "ERROR SUMMARY: 0 errors" not in log:
        raise RuntimeError("Compute Sanitizer racecheck did not report zero errors")
    for operation in (
        "recurrent_rwkv7_packed_stateful_fp32io16",
        "recurrent_rwkv7_packed_stateful_fp16",
    ):
        if operation not in log:
            raise RuntimeError(f"Compute Sanitizer log lacks {operation}")
    if provider_revision not in log:
        raise RuntimeError("Compute Sanitizer log lacks the exact provider revision")
    return {
        "tool": "compute-sanitizer --tool racecheck",
        "log": str(path),
        "zero_errors": True,
        "operations": 2,
    }


def _gpu_gate(args: argparse.Namespace, source_revision: str) -> None:
    _validate_revision("provider revision", args.provider_revision)
    if args.provider_revision != FLASH_RWKV_SOURCE_REVISION:
        raise ValueError(
            "GPU acceptance requires FlashRWKV revision "
            f"{FLASH_RWKV_SOURCE_REVISION!r}"
        )
    if args.runner_label != PRO6000_RUNNER_LABEL:
        raise ValueError(f"GPU acceptance requires runner label {PRO6000_RUNNER_LABEL!r}")

    racecheck = _validate_racecheck(args.racecheck_log, args.provider_revision)
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/ops/test_rwkv7_backends.py",
            "-k",
            "real_provider",
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for dtype in args.dtype:
        output = args.output_dir / f"benchmark-{dtype}.json"
        _run(
            [
                sys.executable,
                "benchmarks/ops/benchmark_rwkv7_flash_provider.py",
                "--dtype",
                dtype,
                "--batch-size",
                str(args.batch_size),
                "--tokens",
                str(args.tokens),
                "--heads",
                str(args.heads),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--runner-label",
                args.runner_label,
                "--pr-number",
                str(args.pr_number),
                "--expected-source-revision",
                source_revision,
                "--expected-provider-revision",
                args.provider_revision,
                "--output",
                str(output),
            ]
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        _validate_benchmark(
            report,
            source_revision=source_revision,
            provider_revision=args.provider_revision,
            runner_label=args.runner_label,
            pr_number=args.pr_number,
            dtype=dtype,
            batch_size=args.batch_size,
            tokens=args.tokens,
            warmup=args.warmup,
            iters=args.iters,
        )
        reports.append(report)

    packed_reports = []
    for mode in ("fp32io16", "fp16"):
        output = args.output_dir / f"packed-benchmark-{mode}.json"
        _run(
            [
                sys.executable,
                "benchmarks/ops/benchmark_rwkv7_flash_packed_provider.py",
                "--mode",
                mode,
                "--profiles",
                "decode_b320",
                "equal16_b320",
                "ragged16_b320",
                "--hidden-size",
                "4096",
                "--warmup",
                "3",
                "--iters",
                "10",
                "--runner-label",
                args.runner_label,
                "--pr-number",
                str(args.pr_number),
                "--expected-source-revision",
                source_revision,
                "--expected-provider-revision",
                args.provider_revision,
                "--output",
                str(output),
            ]
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        _validate_packed_benchmark(
            report,
            source_revision=source_revision,
            provider_revision=args.provider_revision,
            runner_label=args.runner_label,
            pr_number=args.pr_number,
            mode=mode,
        )
        packed_reports.append(report)

    manifest = {
        "schema_version": 1,
        "pr_number": args.pr_number,
        "source_revision": source_revision,
        "flash_rwkv_source_revision": args.provider_revision,
        "runner_label": args.runner_label,
        "correctness_gate": "tests/ops/test_rwkv7_backends.py -k real_provider",
        "racecheck": racecheck,
        "benchmarks": reports,
        "packed_benchmarks": packed_reports,
        "flash_rwkv_clean_evidence": {
            "runtime_semantic_revision": args.provider_revision,
            "evidence_revision": FLASH_RWKV_EVIDENCE_REVISION,
            "workflow_run_id": FLASH_RWKV_EVIDENCE_RUN_ID,
            "artifact_id": FLASH_RWKV_EVIDENCE_ARTIFACT_ID,
            "artifact_digest": FLASH_RWKV_EVIDENCE_ARTIFACT_DIGEST,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("quick", "gpu"))
    parser.add_argument("--source-revision")
    parser.add_argument("--provider-revision")
    parser.add_argument("--runner-label", default="local")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rwkv7-flash-adapter"))
    parser.add_argument("--racecheck-log", type=Path, default=Path("artifacts/rwkv7-flash-adapter/racecheck.log"))
    parser.add_argument("--dtype", action="append", choices=("float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    source_revision = _revision()
    expected_source_revision = args.source_revision or source_revision
    _validate_revision("source revision", expected_source_revision)
    if source_revision != expected_source_revision:
        raise RuntimeError(f"source revision mismatch: expected={expected_source_revision} actual={source_revision}")

    if args.mode == "quick":
        _quick_gate(source_revision)
        return
    if not args.provider_revision:
        parser.error("gpu mode requires --provider-revision")
    if not args.pr_number or args.pr_number <= 0:
        parser.error("gpu mode requires a positive --pr-number")
    args.dtype = args.dtype or ["float16", "bfloat16"]
    if min(args.batch_size, args.tokens, args.heads, args.warmup, args.iters) <= 0:
        parser.error("GPU benchmark shape, warmup, and iteration values must be positive")
    _gpu_gate(args, source_revision)


if __name__ == "__main__":
    main()
