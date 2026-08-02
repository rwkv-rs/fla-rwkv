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


def _tensor(*, dtype=torch.float16, size=64, cuda=True, requires_grad=False, device='cuda:0'):
    return SimpleNamespace(
        dtype=dtype,
        shape=(1, 32, 2, size),
        is_cuda=cuda,
        requires_grad=requires_grad,
        device=torch.device(device),
    )


def _call_args(**overrides):
    args = {name: _tensor() for name in ('r', 'w', 'k', 'v', 'a', 'b')}
    args.update(overrides)
    return args


def test_flash_rwkv_backend_requires_opt_in(monkeypatch):
    monkeypatch.delenv('FLA_FLASH_RWKV', raising=False)
    assert FlashRWKVBackend.is_enabled() is False

    monkeypatch.setenv('FLA_FLASH_RWKV', '1')
    assert FlashRWKVBackend.is_enabled() is True


@pytest.mark.parametrize(
    ('overrides', 'kwargs', 'reason'),
    [
        ({'r': _tensor(cuda=False, device='cpu')}, {}, 'FlashRWKV requires CUDA tensors'),
        ({'r': _tensor(dtype=torch.bfloat16)}, {}, 'FlashRWKV requires float16 inputs'),
        ({'v': _tensor(size=128)}, {}, 'FlashRWKV requires K=V=64, got K=64, V=128'),
        ({'r': _tensor(device='cuda:1')}, {}, 'FlashRWKV requires all inputs on the same CUDA device'),
        ({'r': _tensor(requires_grad=True)}, {'cu_seqlens': _tensor()}, 'FlashRWKV packed execution is forward-only'),
        (
            {'r': _tensor(requires_grad=True)},
            {'initial_state': _tensor(dtype=torch.float16)},
            'FlashRWKV training requires an FP32 initial_state',
        ),
        ({}, {'cp_context': object()}, 'FlashRWKV does not support context parallel execution'),
        ({}, {'disable_recompute': True}, 'FlashRWKV does not support disable_recompute=True'),
        ({}, {'chunk_size': 8}, 'FlashRWKV chunk_size must be 16, 32, or 64, got 8'),
    ],
)
def test_flash_rwkv_verifier_rejection_decision_table(overrides, kwargs, reason):
    accepted, actual_reason = FlashRWKVBackend().chunk_rwkv7_verifier(**_call_args(**overrides), **kwargs)

    assert accepted is False
    assert actual_reason == reason


@pytest.mark.parametrize(
    ('requires_grad', 'packed'),
    [(False, False), (False, True), (True, False)],
)
def test_flash_rwkv_verifier_accepts_supported_execution(requires_grad, packed):
    args = _call_args(r=_tensor(requires_grad=requires_grad))
    accepted, reason = FlashRWKVBackend().chunk_rwkv7_verifier(
        **args,
        cu_seqlens=_tensor() if packed else None,
        initial_state=_tensor(dtype=torch.float32) if requires_grad else None,
        output_final_state=True,
    )

    assert accepted is True
    assert reason is None


def test_chunk_rwkv7_dispatches_to_flash_provider(monkeypatch):
    calls = []
    expected = ('flash-output', 'flash-state')
    fake_provider = SimpleNamespace(rwkv7=lambda *args, **kwargs: calls.append((args, kwargs)) or expected)
    monkeypatch.setitem(sys.modules, 'flash_rwkv', fake_provider)
    monkeypatch.setenv('FLA_FLASH_RWKV', '1')
    monkeypatch.setattr(FlashRWKVBackend, 'is_available', classmethod(lambda cls: True))

    actual = chunk_rwkv7(**_call_args(), output_final_state=True)

    assert actual == expected
    assert get_last_rwkv7_provider() == 'flash_rwkv'
    assert calls[0][1]['algorithm'] == 'chunk'
    assert calls[0][1]['mode'] == 'fp32io16'


def test_chunk_rwkv7_falls_back_to_fla(monkeypatch):
    chunk_module = importlib.import_module('fla.ops.rwkv7.chunk')
    expected = ('fla-output', 'fla-state')
    monkeypatch.setenv('FLA_FLASH_RWKV', '1')
    monkeypatch.setattr(FlashRWKVBackend, 'is_available', classmethod(lambda cls: False))
    monkeypatch.setattr(chunk_module, 'chunk_dplr_delta_rule', lambda **kwargs: expected)

    actual = chunk_rwkv7(**_call_args())

    assert actual == expected
    assert get_last_rwkv7_provider() == 'fla'
