# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from fla.ops.rwkv7 import chunk_rwkv7, get_last_rwkv7_provider
from fla.ops.rwkv7.backends.flash_rwkv import FlashRWKVBackend


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


def test_flash_rwkv_backend_requires_opt_in(monkeypatch):
    monkeypatch.delenv("FLA_FLASH_RWKV", raising=False)
    assert FlashRWKVBackend.is_enabled() is False

    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    assert FlashRWKVBackend.is_enabled() is True


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
            {"cu_seqlens": torch.tensor([0, 32], dtype=torch.int32)},
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
            {"cu_seqlens": torch.tensor([0.0, 32.0])},
            "FlashRWKV cu_seqlens must be a rank-1 int32 or int64 tensor",
        ),
        (
            {},
            {"cu_seqlens": torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32)[::2]},
            "FlashRWKV cu_seqlens must be contiguous",
        ),
        (
            {},
            {"cu_seqlens": torch.tensor([0, 16, 16, 32], dtype=torch.int32)},
            "FlashRWKV cu_seqlens must be strictly increasing",
        ),
        (
            {},
            {
                "cu_seqlens": torch.tensor([0, 12, 32], dtype=torch.int32),
                "initial_state": _tensor(dtype=torch.float32, shape=(1, 2, 64, 64)),
            },
            "FlashRWKV initial_state must have shape [N, H, K, V] matching the input layout",
        ),
        ({}, {"scale": float("nan")}, "FlashRWKV requires a finite scale"),
        ({}, {"safe_gate": True}, "FlashRWKV does not expose the FLA safe_gate contract"),
        ({}, {"cp_context": object()}, "FlashRWKV does not support context parallel execution"),
        ({}, {"disable_recompute": True}, "FlashRWKV does not support disable_recompute=True"),
        ({}, {"chunk_size": 8}, "FlashRWKV chunk_size must be 16, 32, or 64, got 8"),
    ],
)
def test_flash_rwkv_verifier_rejection_decision_table(overrides, kwargs, reason):
    accepted, actual_reason = FlashRWKVBackend().chunk_rwkv7_verifier(**_call_args(**overrides), **kwargs)

    assert accepted is False
    assert actual_reason == reason


@pytest.mark.parametrize(
    ("requires_grad", "packed"),
    [(False, False), (False, True), (True, False)],
)
def test_flash_rwkv_verifier_accepts_supported_execution(requires_grad, packed):
    args = _call_args(r=_tensor(requires_grad=requires_grad))
    cu_seqlens = torch.tensor([0, 12, 32], dtype=torch.int32) if packed else None
    accepted, reason = FlashRWKVBackend().chunk_rwkv7_verifier(
        **args,
        cu_seqlens=cu_seqlens,
        initial_state=_tensor(dtype=torch.float32, shape=(1, 2, 64, 64)) if requires_grad else None,
        output_final_state=True,
    )

    assert accepted is True
    assert reason is None


def test_flash_rwkv_verifier_accepts_packed_state_count_and_cpu_offsets():
    offsets = torch.tensor([0, 12, 32], dtype=torch.int32)
    accepted, reason = FlashRWKVBackend().chunk_rwkv7_verifier(
        **_call_args(),
        cu_seqlens=offsets,
        cu_seqlens_cpu=offsets.clone(),
        initial_state=_tensor(dtype=torch.float32, shape=(2, 2, 64, 64)),
    )

    assert accepted is True
    assert reason is None


def test_flash_rwkv_availability_checks_version_and_public_api(monkeypatch):
    def compatible_rwkv7(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        *,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        mode,
        algorithm,
        chunk_size,
    ):
        del r, log_decay, k, v, a, b, scale, initial_state, output_final_state, cu_seqlens, mode, algorithm, chunk_size

    module = SimpleNamespace(__version__="0.1.0", rwkv7=compatible_rwkv7)
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert FlashRWKVBackend.is_available() is True

    module.__version__ = "0.0.9"
    assert FlashRWKVBackend.is_available() is False
    module.__version__ = "0.1.0"
    module.rwkv7 = lambda: None
    assert FlashRWKVBackend.is_available() is False


def test_chunk_rwkv7_dispatches_to_flash_provider(monkeypatch):
    calls = []
    expected = ("flash-output", "flash-state")
    fake_provider = SimpleNamespace(rwkv7=lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    monkeypatch.setitem(sys.modules, "flash_rwkv", fake_provider)
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: True))

    actual = chunk_rwkv7(**_call_args(), output_final_state=True)

    assert actual == expected
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert calls[0][1]["algorithm"] == "chunk"
    assert calls[0][1]["mode"] == "fp32io16"


def test_explicit_flash_rwkv_unavailable_fails_closed(monkeypatch):
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    monkeypatch.setattr(FlashRWKVBackend, "is_available", classmethod(lambda cls: False))

    with pytest.raises(RuntimeError, match="explicit backend 'flash_rwkv' is unavailable"):
        chunk_rwkv7(**_call_args())
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
        chunk_rwkv7(**_call_args())
    assert get_last_rwkv7_provider() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("batch_size", "sequence_length"), [(1, 16), (2, 32)])
def test_flash_rwkv_real_provider_matches_torch_training_cell(
    monkeypatch,
    batch_size,
    sequence_length,
):
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

    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    actual, actual_state = chunk_rwkv7(
        *inputs,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=16,
    )
    (actual.float().square().mean() + actual_state.square().mean()).backward()

    assert get_last_rwkv7_provider() == "flash_rwkv"
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
