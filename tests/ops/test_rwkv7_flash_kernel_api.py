# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from types import SimpleNamespace

import pytest

import fla.ops
import fla.ops.rwkv7
from fla.ops.rwkv7 import flash_rwkv as flash_api
from fla.ops.rwkv7.backends import flash_rwkv as flash_backend
from fla.ops.rwkv7.backends.provider import (
    get_last_rwkv7_kernel,
    get_last_rwkv7_provider,
    set_last_rwkv7_kernel,
    set_last_rwkv7_provider,
)

EXPECTED_PARAMETERS = {
    "decay_logits_to_log_decay": ("decay_logits",),
    "infer_chunk_bf16_forward": (
        "r", "log_decay", "k", "v", "a", "b", "initial_state", "scale", "output_final_state",
    ),
    "infer_chunk_bf16_forward_varlen": (
        "r", "log_decay", "k", "v", "a", "b", "initial_state", "cu_seqlens", "scale", "output_final_state",
    ),
    "infer_cmix_mix_fp16": ("x", "shift_state", "mix"),
    "infer_recurrent_fp16_forward_varlen": (
        "r", "log_decay", "k", "v", "a", "b", "initial_state", "cu_seqlens", "state_indices", "scale",
        "output_final_state",
    ),
    "infer_recurrent_fp32io16_forward_varlen": (
        "r", "log_decay", "k", "v", "a", "b", "initial_state", "cu_seqlens", "state_indices", "scale",
        "output_final_state",
    ),
    "infer_tmix_kk_a_gate_fp16": ("key", "key_scale", "gate_bias", "gate_delta", "key_gate_scale"),
    "infer_tmix_lnx_rkvres_xg_fp16": (
        "recurrent_output", "receptance", "key", "value", "residual_scale", "norm_weight", "norm_bias", "gate",
    ),
    "infer_tmix_mix6_fp16": ("x", "shift_state", "mixes"),
    "infer_tmix_vres_gate_fp16": ("value", "first_value", "gate_bias", "gate_delta"),
    "pretrain_cmix_bf16": ("x", "x_k", "key_weight", "value_weight"),
    "pretrain_head_l2wrap_ce_bf16": ("hidden", "weight", "targets", "chunk_rows"),
    "pretrain_l2wrap_ce_bf16": ("logits", "targets"),
    "pretrain_recurrent_fp32io16": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state",
    ),
    "pretrain_recurrent_fp32io16_forward": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state",
    ),
    "pretrain_tmix_a_gate_bf16": ("a0", "a12"),
    "pretrain_tmix_kk_pre_bf16": ("key", "key_scale", "learning_rate", "learning_rate_scale"),
    "pretrain_tmix_lnx_rkvres_xg_bf16": (
        "recurrent_output", "receptance", "key", "value", "residual_scale", "norm_weight", "norm_bias", "gate",
    ),
    "pretrain_tmix_mix6_bf16": ("x", "x_r", "x_w", "x_k", "x_v", "x_a", "x_g"),
    "pretrain_tmix_vres_gate_bf16": ("value", "first_value", "v0", "v12"),
    "rwkv7": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state", "cu_seqlens",
        "state_indices", "mode", "algorithm", "chunk_size", "chunk_config",
    ),
    "rwkv7_from_decay_logits": (
        "r", "decay_logits", "k", "v", "a", "b", "scale", "initial_state", "output_final_state",
        "cu_seqlens", "state_indices", "mode", "algorithm", "chunk_size", "chunk_config",
    ),
    "rwkv7_recurrent_stateful": (
        "r", "log_decay", "k", "v", "a", "b", "state_pool", "cu_seqlens", "state_indices", "scale", "mode",
    ),
    "rwkv7_reference": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state", "cu_seqlens",
        "state_indices",
    ),
    "statetune_recurrent_fp32io16_forward": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state",
    ),
}


def test_complete_flash_rwkv_operator_suite_is_public_through_fla():
    assert fla.ops.flash_rwkv is flash_api
    assert fla.ops.rwkv7.flash_rwkv is flash_api
    assert tuple(flash_api.__all__) == flash_backend.FLASH_RWKV_PUBLIC_OPERATORS
    assert set(EXPECTED_PARAMETERS) == set(flash_backend.FLASH_RWKV_PUBLIC_OPERATORS)

    for name, parameters in EXPECTED_PARAMETERS.items():
        operator = getattr(flash_api, name)
        assert callable(operator)
        assert tuple(inspect.signature(operator).parameters) == parameters


def test_every_public_flash_rwkv_operator_routes_to_its_exact_provider_entrypoint(monkeypatch):
    calls = []
    result = object()

    def invoke(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return result

    monkeypatch.setattr(flash_api, "_invoke", invoke)

    for name in flash_backend.FLASH_RWKV_PUBLIC_OPERATORS:
        operator = getattr(flash_api, name)
        args = []
        kwargs = {}
        for parameter in inspect.signature(operator).parameters.values():
            if parameter.default is not inspect.Parameter.empty:
                continue
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = object()
            else:
                args.append(object())
        assert operator(*args, **kwargs) is result
        assert calls[-1][0] == name

    assert [call[0] for call in calls] == list(flash_backend.FLASH_RWKV_PUBLIC_OPERATORS)


def test_provider_namespace_preflights_then_records_exact_operator(monkeypatch):
    calls = []
    expected = object()
    provider = SimpleNamespace(
        decay_logits_to_log_decay=lambda value: calls.append(("provider", value)) or expected
    )
    monkeypatch.setattr(flash_api, "preflight_flash_rwkv_installation", lambda: calls.append(("preflight",)))
    monkeypatch.setattr(
        flash_api.importlib,
        "import_module",
        lambda name: calls.append(("import", name)) or provider,
    )

    value = object()
    assert flash_api.decay_logits_to_log_decay(value) is expected
    assert calls == [("preflight",), ("import", "flash_rwkv"), ("provider", value)]
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert get_last_rwkv7_kernel() == "decay_logits_to_log_decay"


def test_provider_namespace_fails_closed_before_import(monkeypatch):
    set_last_rwkv7_provider("stale")
    set_last_rwkv7_kernel("stale")
    monkeypatch.setattr(
        flash_api,
        "preflight_flash_rwkv_installation",
        lambda: (_ for _ in ()).throw(flash_backend.FlashRWKVProvenanceError("wrong revision")),
    )
    monkeypatch.setattr(
        flash_api.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected provider import: {name}"),
    )

    with pytest.raises(flash_backend.FlashRWKVProvenanceError, match="wrong revision"):
        flash_api.decay_logits_to_log_decay(object())

    assert get_last_rwkv7_provider() is None
    assert get_last_rwkv7_kernel() is None
