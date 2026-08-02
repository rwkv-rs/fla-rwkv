# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import inspect
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.ops.benchmark_rwkv7_flash_provider import _format_result
from fla.ops.rwkv7 import get_last_rwkv7_provider, recurrent_rwkv7
from fla.ops.rwkv7.backends import flash_rwkv as flash_rwkv_backend
from fla.ops.rwkv7.backends.flash_rwkv import (
    FLASH_RWKV_SOURCE_REVISION,
    FlashRWKVBackend,
    FlashRWKVProvenanceError,
)
from scripts.run_rwkv7_flash_adapter_ci import _validate_benchmark


def _tensor(
    *,
    dtype=torch.float16,
    size=64,
    cuda=True,
    requires_grad=False,
    device="cuda:0",
    shape=None,
    contiguous=True,
):
    shape = shape or (1, 32, 2, size)
    return SimpleNamespace(
        dtype=dtype,
        shape=shape,
        ndim=len(shape),
        is_cuda=cuda,
        requires_grad=requires_grad,
        device=torch.device(device),
        is_contiguous=lambda: contiguous,
        is_floating_point=lambda: dtype.is_floating_point,
    )


def _call_args(**overrides):
    args = {name: _tensor() for name in ("r", "w", "k", "v", "a", "b")}
    args.update(overrides)
    return args


def _metadata(values, *, dtype=torch.int32, device="cuda:0", contiguous=True):
    return SimpleNamespace(
        values=tuple(values),
        dtype=dtype,
        shape=(len(values),),
        ndim=1,
        device=torch.device(device),
        is_contiguous=lambda: contiguous,
    )


def test_flash_rwkv_backend_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    assert FlashRWKVBackend.is_enabled() is True

    monkeypatch.setenv("FLA_FLASH_RWKV", "0")
    assert FlashRWKVBackend.is_enabled() is False


def test_public_stateful_signature_matches_vllm_consumer_contract():
    required = {"initial_state", "output_final_state", "cu_seqlens", "state_indices", "mode"}

    assert required <= inspect.signature(recurrent_rwkv7).parameters.keys()
    assert required <= inspect.signature(FlashRWKVBackend.recurrent_rwkv7).parameters.keys()
    assert required <= inspect.signature(FlashRWKVBackend.recurrent_rwkv7_verifier).parameters.keys()
    assert tuple(inspect.signature(recurrent_rwkv7).parameters) == (
        "r",
        "w",
        "k",
        "v",
        "a",
        "b",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "cu_seqlens_cpu",
        "state_indices",
        "mode",
        "safe_gate",
        "chunk_size",
        "disable_recompute",
        "cp_context",
        "kwargs",
    )


def test_flash_rwkv_verifier_does_not_materialize_device_metadata_on_host():
    source = inspect.getsource(FlashRWKVBackend.recurrent_rwkv7_verifier)

    assert ".cpu(" not in source
    assert ".tolist(" not in source


def test_public_contract_owns_only_exact_recurrent_provider():
    import fla.ops
    import fla.ops.rwkv7

    implementation = inspect.getsource(FlashRWKVBackend.recurrent_rwkv7)
    layer = (Path(__file__).parents[2] / "fla/layers/rwkv7.py").read_text(encoding="utf-8")

    assert fla.ops.recurrent_rwkv7 is recurrent_rwkv7
    assert fla.ops.rwkv7.recurrent_rwkv7 is recurrent_rwkv7
    assert not hasattr(fla.ops, "chunk_rwkv7_reference")
    assert not hasattr(fla.ops.rwkv7, "chunk_rwkv7_reference")
    assert 'algorithm="recurrent"' in implementation
    assert "chunk_rwkv7_reference" not in implementation
    assert "chunk_dplr_delta_rule" not in implementation
    assert "pretrain_recurrent_fp32io16_forward" in implementation
    assert "rwkv7_recurrent_stateful" in implementation
    assert "if mode == 'recurrent':" in layer
    assert "return recurrent_rwkv7(" in layer


def test_rwkv7_model_defaults_to_recurrent_product_execution():
    from fla.layers.rwkv7 import RWKV7Attention
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config

    assert RWKV7Config().attn_mode == "recurrent"
    assert inspect.signature(RWKV7Attention).parameters["mode"].default == "recurrent"


@pytest.mark.parametrize(
    ("mode", "selected"),
    [
        ("recurrent", "recurrent"),
        ("chunk", "explicit_chunk_reference"),
        ("fused_recurrent", "fused_recurrent"),
    ],
)
def test_rwkv7_layer_explicit_mode_selects_exact_operator(monkeypatch, mode, selected):
    import fla.layers.rwkv7 as layer

    calls = []

    def implementation(name):
        def run(**kwargs):
            calls.append((name, kwargs))
            return name, None

        return run

    monkeypatch.setattr(layer, "recurrent_rwkv7", implementation("recurrent"))
    monkeypatch.setattr(layer, "chunk_rwkv7_reference", implementation("explicit_chunk_reference"))
    monkeypatch.setattr(layer, "fused_mul_recurrent_rwkv7", implementation("fused_recurrent"))
    tensor = torch.ones(1, 1, 1, 1)

    actual = layer._run_rwkv7_operator(
        mode,
        r=tensor,
        w=tensor,
        k=tensor,
        v=tensor,
        kk=tensor,
        a=tensor,
        recurrent_state=None,
        output_final_state=False,
        cu_seqlens=None,
    )

    assert actual == (selected, None)
    assert [name for name, _ in calls] == [selected]
    if mode == "chunk":
        assert calls[0][1]["safe_gate"] is True
        assert calls[0][1]["chunk_size"] == 64


def test_rwkv7_layer_rejects_unknown_explicit_mode():
    from fla.layers.rwkv7 import RWKV7Attention

    with pytest.raises(ValueError, match="Not supported mode `unknown`"):
        RWKV7Attention(mode="unknown")


@pytest.mark.parametrize(
    ("overrides", "kwargs", "reason"),
    [
        ({"r": _tensor(cuda=False, device="cpu")}, {}, "FlashRWKV requires CUDA tensors"),
        ({"r": _tensor(dtype=torch.float32)}, {}, "FlashRWKV requires float16 or bfloat16 inputs"),
        ({"r": _tensor(shape=(1, 32, 128))}, {}, "FlashRWKV requires rank-4 [B, T, H, D] inputs"),
        ({"r": _tensor(contiguous=False)}, {}, "FlashRWKV requires contiguous inputs"),
        ({"v": _tensor(size=128)}, {}, "FlashRWKV requires K=V=64, got K=64, V=128"),
        ({"r": _tensor(device="cuda:1")}, {}, "FlashRWKV requires all inputs on the same CUDA device"),
        (
            {"r": _tensor(requires_grad=True)},
            {"cu_seqlens": _metadata((0, 32))},
            "FlashRWKV packed execution is forward-only",
        ),
        (
            {"r": _tensor(requires_grad=True)},
            {"initial_state": _tensor(dtype=torch.float16, shape=(1, 2, 64, 64))},
            "FlashRWKV training requires an FP32 initial_state",
        ),
        (
            {},
            {"initial_state": _tensor(dtype=torch.float32, shape=(2, 2, 64, 64))},
            "FlashRWKV initial_state must have shape [N, H, K, V] matching the input layout",
        ),
        (
            {},
            {"initial_state": _tensor(dtype=torch.int32, shape=(1, 2, 64, 64))},
            "FlashRWKV initial_state must have a floating-point dtype",
        ),
        (
            {},
            {"cu_seqlens": _metadata((0, 32), dtype=torch.float32)},
            "FlashRWKV cu_seqlens must be a rank-1 int32 tensor",
        ),
        (
            {},
            {"cu_seqlens": _metadata((0, 16, 32), contiguous=False)},
            "FlashRWKV cu_seqlens must be contiguous",
        ),
        (
            {},
            {"cu_seqlens": _metadata((0, 32), device="cpu")},
            "FlashRWKV cu_seqlens must be on the input CUDA device",
        ),
        (
            {},
            {"cu_seqlens": _metadata((0,))},
            "FlashRWKV cu_seqlens must describe at least one sequence",
        ),
        (
            {},
            {
                "cu_seqlens": _metadata((0, 12, 32)),
                "initial_state": _tensor(dtype=torch.float32, shape=(1, 2, 64, 64)),
            },
            "FlashRWKV initial_state must have shape [N, H, K, V] matching the input layout",
        ),
        (
            {},
            {
                "cu_seqlens": _metadata((0, 32)),
                "cu_seqlens_cpu": _metadata((0, 32), device="cpu"),
            },
            "FlashRWKV recurrent API does not accept duplicate CPU packed metadata",
        ),
        ({}, {"scale": float("nan")}, "FlashRWKV requires a finite scale"),
        ({}, {"safe_gate": True}, "FlashRWKV recurrent API does not expose the FLA safe_gate contract"),
        ({}, {"cp_context": object()}, "FlashRWKV recurrent API does not support context parallel execution"),
        ({}, {"disable_recompute": True}, "FlashRWKV recurrent API does not support disable_recompute=True"),
        ({}, {"chunk_size": 8}, "FlashRWKV recurrent API does not accept chunk_size"),
    ],
)
def test_flash_rwkv_verifier_rejection_decision_table(overrides, kwargs, reason):
    accepted, actual_reason = FlashRWKVBackend().recurrent_rwkv7_verifier(**_call_args(**overrides), **kwargs)

    assert accepted is False
    assert actual_reason == reason


@pytest.mark.parametrize(
    ("requires_grad", "packed"),
    [(False, False), (False, True), (True, False)],
)
def test_flash_rwkv_verifier_accepts_supported_execution(requires_grad, packed):
    args = _call_args(r=_tensor(requires_grad=requires_grad))
    cu_seqlens = _metadata((0, 12, 32)) if packed else None
    accepted, reason = FlashRWKVBackend().recurrent_rwkv7_verifier(
        **args,
        cu_seqlens=cu_seqlens,
        initial_state=_tensor(dtype=torch.float32, shape=(1, 2, 64, 64)) if requires_grad else None,
        output_final_state=True,
    )

    assert accepted is True
    assert reason is None


def test_flash_rwkv_verifier_accepts_packed_state_count():
    offsets = _metadata((0, 12, 32))
    accepted, reason = FlashRWKVBackend().recurrent_rwkv7_verifier(
        **_call_args(),
        cu_seqlens=offsets,
        initial_state=_tensor(dtype=torch.float32, shape=(2, 2, 64, 64)),
    )

    assert accepted is True
    assert reason is None


@pytest.mark.parametrize(
    (
        "state_indices",
        "state_shape",
        "mode",
        "state_dtype",
        "output_final_state",
        "reason",
    ),
    [
        (
            _metadata((3,)),
            (4, 2, 64, 64),
            "fp32io16",
            torch.float32,
            True,
            "FlashRWKV state_indices length must match the packed sequence count",
        ),
        (
            _metadata((3, 1), dtype=torch.int64),
            (4, 2, 64, 64),
            "fp32io16",
            torch.float32,
            True,
            "FlashRWKV state_indices must be a rank-1 int32 tensor",
        ),
        (
            _metadata((3, 1), device="cpu"),
            (4, 2, 64, 64),
            "fp32io16",
            torch.float32,
            True,
            "FlashRWKV state_indices must be on the input CUDA device",
        ),
        (
            _metadata((3, 1)),
            (4, 2, 64, 64),
            "fp32io16",
            torch.float32,
            False,
            "FlashRWKV stateful packed execution requires output_final_state=True",
        ),
        (
            _metadata((3, 1)),
            (4, 2, 32, 64),
            "fp32io16",
            torch.float32,
            True,
            "FlashRWKV initial_state must have shape [N, H, K, V] matching the input layout",
        ),
        (
            _metadata((3, 1)),
            (4, 2, 64, 64),
            "fp16",
            torch.float32,
            True,
            "FlashRWKV stateful fp16 requires a torch.float16 state pool",
        ),
    ],
)
def test_flash_rwkv_verifier_rejects_invalid_state_pool_slots(
    state_indices,
    state_shape,
    mode,
    state_dtype,
    output_final_state,
    reason,
):
    accepted, actual_reason = FlashRWKVBackend().recurrent_rwkv7_verifier(
        **{
            name: _tensor(shape=(1, 3, 2, 64))
            for name in ("r", "w", "k", "v", "a", "b")
        },
        cu_seqlens=_metadata((0, 2, 3)),
        state_indices=state_indices,
        initial_state=_tensor(dtype=state_dtype, shape=state_shape),
        output_final_state=output_final_state,
        mode=mode,
    )

    assert accepted is False
    assert actual_reason == reason


@pytest.mark.parametrize(("mode", "state_dtype"), [("fp16", torch.float16), ("fp32io16", torch.float32)])
def test_flash_rwkv_verifier_accepts_mixed_wave_noncontiguous_slots(mode, state_dtype):
    accepted, reason = FlashRWKVBackend().recurrent_rwkv7_verifier(
        **{
            name: _tensor(shape=(1, 3, 2, 64))
            for name in ("r", "w", "k", "v", "a", "b")
        },
        cu_seqlens=_metadata((0, 2, 3)),
        state_indices=_metadata((3, 1)),
        initial_state=_tensor(dtype=state_dtype, shape=(4, 2, 64, 64)),
        output_final_state=True,
        mode=mode,
    )

    assert accepted is True
    assert reason is None


def test_flash_rwkv_availability_revalidates_public_provenance(monkeypatch):
    results = [object(), FlashRWKVProvenanceError("dirty editable checkout")]

    def validate():
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(flash_rwkv_backend, "validate_flash_rwkv_installation", validate)

    assert FlashRWKVBackend.is_available() is True
    assert FlashRWKVBackend.is_available() is False
    assert results == []


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("dirty-editable", "checkout is dirty"),
        ("wrong-origin", "origin is not"),
        ("shadow-module", "module is not owned"),
        ("wrong-native", "_C is not owned"),
        ("no-direct-url", "lacks PEP 610 direct_url.json"),
        ("wrong-revision", "PEP 610 commit must be"),
    ],
)
def test_flash_rwkv_provenance_fresh_process_negatives(
    tmp_path,
    scenario,
    expected,
):
    program = r'''
import importlib.machinery
import json
import os
from pathlib import Path
from types import SimpleNamespace

import fla.ops.rwkv7.backends.flash_rwkv as backend

root = Path(os.environ["CASE_ROOT"])
scenario = os.environ["CASE_SCENARIO"]
editable = scenario in {"dirty-editable", "wrong-origin"}
source_root = root / ("editable" if editable else "installed")
package_root = source_root / "flash_rwkv"
package_root.mkdir(parents=True)
module_root = root / "shadow" / "flash_rwkv" if scenario == "shadow-module" else package_root
module_root.mkdir(parents=True, exist_ok=True)
module_path = module_root / "__init__.py"
module_path.write_text("", encoding="utf-8")
native_root = root / "foreign" if scenario == "wrong-native" else package_root
native_root.mkdir(parents=True, exist_ok=True)
native_path = native_root / f"_C{importlib.machinery.EXTENSION_SUFFIXES[0]}"
native_path.write_bytes(b"native-placeholder")

def rwkv7(
    r, log_decay, k, v, a, b, *, scale, initial_state,
    output_final_state, cu_seqlens, mode, algorithm, chunk_size,
):
    pass

def rwkv7_recurrent_stateful(
    r, log_decay, k, v, a, b, *, state_pool, cu_seqlens,
    state_indices, scale, mode,
):
    pass

def pretrain_recurrent_fp32io16_forward(
    r, log_decay, k, v, a, b, *, scale, initial_state, output_final_state,
):
    pass

native = SimpleNamespace(__file__=str(native_path))
module = SimpleNamespace(
    __file__=str(module_path),
    __version__="0.1.0",
    _C=native,
    rwkv7=rwkv7,
    rwkv7_recurrent_stateful=rwkv7_recurrent_stateful,
    pretrain_recurrent_fp32io16_forward=pretrain_recurrent_fp32io16_forward,
    validate_packed_metadata_strict=lambda *args, **kwargs: None,
)
direct_url = (
    {"url": source_root.as_uri(), "dir_info": {"editable": True}}
    if editable
    else {
        "url": backend.FLASH_RWKV_REPOSITORY,
        "vcs_info": {
            "vcs": "git",
            "commit_id": (
                "0" * 40
                if scenario == "wrong-revision"
                else backend.FLASH_RWKV_SOURCE_REVISION
            ),
        },
    }
)

class Distribution:
    def read_text(self, name):
        if scenario == "no-direct-url":
            return None
        return json.dumps(direct_url) if name == "direct_url.json" else None

    def locate_file(self, name):
        assert name == "flash_rwkv"
        return package_root

backend.importlib.metadata.distribution = lambda name: Distribution()
backend.importlib.import_module = (
    lambda name: module if name == "flash_rwkv" else native
)
if editable:
    def git_output(repository, *arguments):
        responses = {
            ("rev-parse", "--show-toplevel"): str(source_root),
            ("remote", "get-url", "origin"): (
                "https://example.com/not-flash-rwkv.git"
                if scenario == "wrong-origin"
                else backend.FLASH_RWKV_REPOSITORY
            ),
            ("rev-parse", "HEAD"): backend.FLASH_RWKV_SOURCE_REVISION,
            ("status", "--porcelain=v1", "--untracked-files=all"): (
                " M flash_rwkv/ops.py" if scenario == "dirty-editable" else ""
            ),
        }
        return responses[arguments]
    backend._git_output = git_output

try:
    backend.validate_flash_rwkv_installation()
except backend.FlashRWKVProvenanceError as error:
    if os.environ["EXPECTED_ERROR"] not in str(error):
        raise
else:
    raise AssertionError("hostile provenance was accepted")
'''
    environment = dict(os.environ)
    environment.update(
        CASE_ROOT=str(tmp_path),
        CASE_SCENARIO=scenario,
        EXPECTED_ERROR=expected,
        CUDA_VISIBLE_DEVICES="",
    )
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
    )


def test_pinned_revision_matches_ci_contract():
    root = Path(__file__).parents[2]
    script = (root / "scripts/run_rwkv7_flash_adapter_ci.py").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/rwkv7-flash-adapter.yml").read_text(encoding="utf-8")

    assert f'FLASH_RWKV_SOURCE_REVISION = "{FLASH_RWKV_SOURCE_REVISION}"' in script
    assert f"FLASH_RWKV_SOURCE_REVISION: {FLASH_RWKV_SOURCE_REVISION}" in workflow


def test_benchmark_result_has_complete_stable_fields():
    row = {
        "label": "flash-rwkv-float16-B2T4",
        "B": 2,
        "T": 4,
        "iters": 3,
        "p10_ms": 1.2000000000000002,
        "p50_ms": 2.0,
        "p90_ms": 2.8000000000000003,
        "tok_s_p50": 4000.0,
    }

    assert _format_result(row) == (
        "RESULT B=2 T=4 iters=3 p10_ms=1.2 p50_ms=2.0 "
        "p90_ms=2.8 tok_s_p50=4000.0 label=flash-rwkv-float16-B2T4"
    )
    with pytest.raises(ValueError, match="missing RESULT fields"):
        _format_result({"label": "incomplete"})


def test_packed_benchmark_help_does_not_import_optional_provider(tmp_path):
    root = Path(__file__).parents[2]
    (tmp_path / "flash_rwkv.py").write_text(
        'raise RuntimeError("optional provider imported during CLI discovery")\n',
        encoding="utf-8",
    )
    (tmp_path / "sitecustomize.py").write_text(
        "import importlib.metadata as metadata\n"
        "original_distribution = metadata.distribution\n"
        "def distribution(name):\n"
        "    if name == 'flash-rwkv':\n"
        "        raise metadata.PackageNotFoundError(name)\n"
        "    return original_distribution(name)\n"
        "metadata.distribution = distribution\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(root)))

    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/ops/benchmark_rwkv7_flash_packed_provider.py"),
            "--help",
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_gpu_gate_validates_complete_result_contract():
    source_revision = "1" * 40
    report = {
        "source_revision": source_revision,
        "pr_number": 7,
        "flash_rwkv_source_revision": FLASH_RWKV_SOURCE_REVISION,
        "backend": "flash_rwkv",
        "selected_provider": "flash_rwkv",
        "oracle": "explicit-pytorch-sequential-recurrent-autograd",
        "dtype": "float16",
        "label": "flash-rwkv-float16-B2T4",
        "B": 2,
        "T": 4,
        "warmup": 1,
        "iters": 3,
        "p10_ms": 1.2,
        "p50_ms": 2.0,
        "p90_ms": 2.8,
        "tok_s_p50": 4000.0,
        "latency_ms": {"p10": 1.2, "p50": 2.0, "p90": 2.8},
        "tokens_per_second": 4000.0,
        "hardware": {
            "runner_label": "rwkv-sha-pro6000x8",
            "device_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        },
        "output_error": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0},
        "final_state_error": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0},
        "input_gradient_error": [
            {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0}
            for _ in range(6)
        ],
        "initial_state_gradient_error": {"max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0},
    }

    _validate_benchmark(
        report,
        source_revision=source_revision,
        provider_revision=FLASH_RWKV_SOURCE_REVISION,
        runner_label="rwkv-sha-pro6000x8",
        pr_number=7,
        dtype="float16",
        batch_size=2,
        tokens=4,
        warmup=1,
        iters=3,
    )
    del report["tok_s_p50"]
    with pytest.raises(RuntimeError, match="lacks RESULT fields"):
        _validate_benchmark(
            report,
            source_revision=source_revision,
            provider_revision=FLASH_RWKV_SOURCE_REVISION,
            runner_label="rwkv-sha-pro6000x8",
            pr_number=7,
            dtype="float16",
            batch_size=2,
            tokens=4,
            warmup=1,
            iters=3,
        )


def test_backend_import_preserves_existing_transformers_rwkv7_config():
    program = """
from transformers import AutoConfig, PretrainedConfig

try:
    native_config = type(AutoConfig.for_model("rwkv7"))
except ValueError:
    class NativeRWKV7Config(PretrainedConfig):
        model_type = "rwkv7"

    AutoConfig.register("rwkv7", NativeRWKV7Config)
    native_config = NativeRWKV7Config

import fla.ops.rwkv7.backends.flash_rwkv

assert type(AutoConfig.for_model("rwkv7")) is native_config
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
    )


def test_recurrent_rwkv7_dispatches_to_flash_provider(monkeypatch):
    calls = []
    expected = ("flash-output", "flash-state")
    fake_provider = SimpleNamespace(rwkv7=lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    monkeypatch.setitem(sys.modules, "flash_rwkv", fake_provider)
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))

    actual = recurrent_rwkv7(**_call_args(), output_final_state=True)

    assert actual == expected
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert calls[0][1]["algorithm"] == "recurrent"
    assert calls[0][1]["mode"] == "fp32io16"


def test_recurrent_rwkv7_dispatches_gradients_to_exact_recurrent_autograd(monkeypatch):
    calls = []
    expected = ("training-output", "training-state")
    fake_provider = SimpleNamespace(
        pretrain_recurrent_fp32io16_forward=lambda *args, **kwargs: calls.append((args, kwargs)) or expected
    )
    monkeypatch.setitem(sys.modules, "flash_rwkv", fake_provider)
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))

    actual = recurrent_rwkv7(
        **_call_args(r=_tensor(requires_grad=True)),
        output_final_state=True,
    )

    assert actual == expected
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert len(calls) == 1
    assert calls[0][1]["output_final_state"] is True


def test_public_recurrent_verifier_rejection_fails_closed(monkeypatch):
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))

    with pytest.raises(RuntimeError, match="does not expose the FLA safe_gate contract"):
        recurrent_rwkv7(**_call_args(), safe_gate=True)
    assert get_last_rwkv7_provider() is None


def test_public_recurrent_dispatch_disabled_fails_closed():
    program = r'''
import os
os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"

import torch
from fla.ops.rwkv7 import recurrent_rwkv7
from fla.ops.rwkv7.backends.flash_rwkv import FlashRWKVBackend

FlashRWKVBackend.is_available = classmethod(lambda cls: True)
FlashRWKVBackend.recurrent_rwkv7_verifier = lambda self, *args, **kwargs: (True, None)
x = torch.zeros(1, 1, 1, 64)
try:
    recurrent_rwkv7(x, x, x, x, x, x)
except RuntimeError as error:
    assert "backend dispatch was bypassed" in str(error)
else:
    raise AssertionError("dispatch-disabled recurrent call did not fail closed")
'''
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
    )


@pytest.mark.parametrize(("mode", "state_dtype"), [("fp16", torch.float16), ("fp32io16", torch.float32)])
def test_public_recurrent_rwkv7_runs_mixed_wave_in_place_without_fallback(
    monkeypatch,
    mode,
    state_dtype,
):
    calls = []
    state_pool = _tensor(dtype=state_dtype, shape=(4, 2, 64, 64))
    state_pool.updated_slots = None
    cu_seqlens = _metadata((0, 2, 3))
    state_indices = _metadata((3, 1))

    def rwkv7_recurrent_stateful(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["state_pool"].updated_slots = kwargs["state_indices"].values
        return "mixed-wave-output"

    fake_provider = SimpleNamespace(rwkv7_recurrent_stateful=rwkv7_recurrent_stateful)
    monkeypatch.setitem(sys.modules, "flash_rwkv", fake_provider)
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))

    inputs = {
        name: _tensor(shape=(1, 3, 2, 64))
        for name in ("r", "w", "k", "v", "a", "b")
    }
    output, final_state = recurrent_rwkv7(
        **inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
    )

    assert output == "mixed-wave-output"
    assert final_state is state_pool
    assert state_pool.updated_slots == (3, 1)
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["state_pool"] is state_pool
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["mode"] == mode


def test_public_stateful_provider_failure_clears_telemetry(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("stateful provider failed")

    monkeypatch.setitem(
        sys.modules,
        "flash_rwkv",
        SimpleNamespace(rwkv7_recurrent_stateful=fail),
    )
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))
    set_provider = importlib.import_module("fla.ops.rwkv7.backends.provider").set_last_rwkv7_provider
    set_provider("flash_rwkv")

    with pytest.raises(RuntimeError, match="stateful provider failed"):
        recurrent_rwkv7(
            **{
                name: _tensor(shape=(1, 3, 2, 64))
                for name in ("r", "w", "k", "v", "a", "b")
            },
            initial_state=_tensor(dtype=torch.float32, shape=(4, 2, 64, 64)),
            output_final_state=True,
            cu_seqlens=_metadata((0, 2, 3)),
            state_indices=_metadata((3, 1)),
            mode="fp32io16",
        )
    assert get_last_rwkv7_provider() is None


def test_explicit_flash_rwkv_unavailable_fails_closed(monkeypatch):
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="explicit backend 'flash_rwkv' is unavailable"):
        recurrent_rwkv7(**_call_args())
    assert get_last_rwkv7_provider() is None


def test_default_flash_rwkv_unavailable_has_no_reference_fallback(monkeypatch):
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        recurrent_rwkv7(**_call_args())
    assert get_last_rwkv7_provider() is None


def test_failed_provider_call_clears_stale_success(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    fake_provider = SimpleNamespace(rwkv7=fail)
    monkeypatch.setitem(sys.modules, "flash_rwkv", fake_provider)
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))
    set_provider = importlib.import_module("fla.ops.rwkv7.backends.provider").set_last_rwkv7_provider
    set_provider("flash_rwkv")

    with pytest.raises(RuntimeError, match="provider failed"):
        recurrent_rwkv7(**_call_args())
    assert get_last_rwkv7_provider() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 16), (2, 32)])
def test_flash_rwkv_real_provider_matches_torch_training_cell(
    monkeypatch,
    batch_size,
    sequence_length,
):
    import flash_rwkv

    torch.manual_seed(7)
    dtype = torch.bfloat16
    inputs = [
        (
            torch.randn(
                batch_size,
                sequence_length,
                1,
                64,
                device="cuda",
                dtype=dtype,
            )
            * 0.02
        )
        .contiguous()
        .requires_grad_()
        for _ in range(6)
    ]
    inputs[1] = torch.full_like(inputs[1], -0.1, requires_grad=True)
    initial_state = (
        torch.randn(
            batch_size, 1, 64, 64, device="cuda", dtype=torch.float32
        )
        * 0.01
    ).requires_grad_()

    baseline_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    baseline_state = initial_state.detach().clone().requires_grad_()
    expected, expected_state = _torch_rwkv7(
        *baseline_inputs, initial_state=baseline_state
    )
    (expected.float().square().mean() + expected_state.square().mean()).backward()

    recurrent_autograd_calls = []
    original = flash_rwkv.pretrain_recurrent_fp32io16_forward

    @wraps(original)
    def observe_recurrent_autograd(*args, **kwargs):
        recurrent_autograd_calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        flash_rwkv,
        "pretrain_recurrent_fp32io16_forward",
        observe_recurrent_autograd,
    )
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    actual, actual_state = recurrent_rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
    )
    (actual.float().square().mean() + actual_state.square().mean()).backward()

    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert len(recurrent_autograd_calls) == 1
    assert actual.dtype == torch.bfloat16
    assert actual_state.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=2e-3)
    for actual_input, expected_input in zip(inputs, baseline_inputs, strict=True):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=5e-2, atol=5e-3
        )
    torch.testing.assert_close(
        initial_state.grad, baseline_state.grad, rtol=5e-2, atol=5e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("mode", "state_dtype"),
    [("fp32io16", torch.float32), ("fp16", torch.float16)],
)
def test_flash_rwkv_real_provider_packed_state_pool_contract(
    monkeypatch,
    mode,
    state_dtype,
):
    import flash_rwkv

    torch.manual_seed(19)
    shape = (1, 3, 1, 64)
    inputs = [
        (torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02).contiguous()
        for _ in range(6)
    ]
    inputs[1] = torch.full_like(inputs[1], -0.1)
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    initial_pool = (
        torch.randn(5, 1, 64, 64, device="cuda", dtype=torch.float32) * 0.01
    ).to(state_dtype)
    expected_output, expected_pool = _torch_rwkv7_packed(
        *inputs,
        initial_state=initial_pool,
        sequence_ranges=((0, 2), (2, 3)),
        state_slots=(3, 1),
    )
    state_pool = initial_pool.clone()
    untouched_slots = torch.tensor([0, 2, 4], device="cuda")
    untouched_before = state_pool.index_select(0, untouched_slots).clone()
    cu_seqlens_pointer = cu_seqlens.data_ptr()
    state_indices_pointer = state_indices.data_ptr()
    observed = {}
    original = flash_rwkv.rwkv7_recurrent_stateful

    @wraps(original)
    def observe_metadata(*args, **kwargs):
        observed["state_pool"] = kwargs["state_pool"]
        observed["cu_seqlens"] = kwargs["cu_seqlens"]
        observed["state_indices"] = kwargs["state_indices"]
        return original(*args, **kwargs)

    monkeypatch.setattr(flash_rwkv, "rwkv7_recurrent_stateful", observe_metadata)
    flash_rwkv.validate_packed_metadata_strict(
        cu_seqlens,
        state_indices,
        total_tokens=shape[1],
        state_pool_size=state_pool.shape[0],
    )
    actual_output, final_state = recurrent_rwkv7(
        *inputs,
        initial_state=state_pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode=mode,
    )
    torch.cuda.synchronize()

    assert final_state is state_pool
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert observed["state_pool"] is state_pool
    assert observed["cu_seqlens"] is cu_seqlens
    assert observed["state_indices"] is state_indices
    assert cu_seqlens.data_ptr() == cu_seqlens_pointer
    assert state_indices.data_ptr() == state_indices_pointer
    _assert_relative_rmse(actual_output, expected_output, maximum=0.003)
    _assert_relative_rmse(
        state_pool.index_select(0, state_indices.long()),
        expected_pool.index_select(0, state_indices.long()),
        maximum=0.003,
    )
    assert torch.equal(
        state_pool.index_select(0, untouched_slots),
        untouched_before,
    )

    graph_pool = initial_pool.clone()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            recurrent_rwkv7(
                *inputs,
                initial_state=graph_pool,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                mode=mode,
            )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_state = recurrent_rwkv7(
            *inputs,
            initial_state=graph_pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert captured_state is graph_pool
    assert observed["cu_seqlens"] is cu_seqlens
    assert observed["state_indices"] is state_indices
    assert torch.isfinite(captured_output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_rwkv_real_provider_strict_debug_validation_rejects_hostile_metadata():
    import flash_rwkv

    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="must be unique"):
        flash_rwkv.validate_packed_metadata_strict(
            cu_seqlens,
            torch.tensor([1, 1], device="cuda", dtype=torch.int32),
            total_tokens=3,
            state_pool_size=4,
        )
    with pytest.raises(ValueError, match="within the state pool"):
        flash_rwkv.validate_packed_metadata_strict(
            cu_seqlens,
            torch.tensor([3, 4], device="cuda", dtype=torch.int32),
            total_tokens=3,
            state_pool_size=4,
        )


def _torch_rwkv7(r, w, k, v, a, b, *, initial_state, scale=1.0):
    state = initial_state.float()
    outputs = []
    for index in range(r.shape[1]):
        state = (
            w[:, index].float().exp().unsqueeze(-1) * state
            + b[:, index].float().unsqueeze(-1)
            * torch.einsum("bhk,bhkv->bhv", a[:, index].float(), state).unsqueeze(-2)
            + k[:, index].float().unsqueeze(-1)
            * v[:, index].float().unsqueeze(-2)
        )
        outputs.append(
            torch.einsum(
                "bhk,bhkv->bhv", r[:, index].float() * scale, state
            ).to(v.dtype)
        )
    return torch.stack(outputs, dim=1), state


def _torch_rwkv7_packed(
    r,
    w,
    k,
    v,
    a,
    b,
    *,
    initial_state,
    sequence_ranges,
    state_slots,
    scale=1.0,
):
    state_pool = initial_state.float().clone()
    output = torch.empty_like(v)
    for (start, end), slot in zip(sequence_ranges, state_slots, strict=True):
        sequence_output, sequence_state = _torch_rwkv7(
            r[:, start:end],
            w[:, start:end],
            k[:, start:end],
            v[:, start:end],
            a[:, start:end],
            b[:, start:end],
            initial_state=state_pool[slot : slot + 1],
            scale=scale,
        )
        output[:, start:end] = sequence_output
        state_pool[slot] = sequence_state[0]
    return output, state_pool


def _assert_relative_rmse(actual, expected, *, maximum):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    baseline = expected.float().square().mean().sqrt().clamp_min(1e-8)
    assert (error / baseline).item() <= maximum
