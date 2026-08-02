# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from types import SimpleNamespace

import pytest
import torch

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
    "rl_infctx_chunk_fp32io16_factor_recompute": (
        "r", "log_decay", "k", "v", "a", "b", "scale", "initial_state", "output_final_state", "cu_seqlens",
        "state_indices", "chunk_size",
    ),
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


def test_standard_attention_inference_routes_all_compatible_fused_blocks(monkeypatch):
    import fla.layers.rwkv7 as layer_module

    calls = []
    recurrence = {}

    def mix6(x, shift_state, mixes):
        calls.append("mix6")
        shift_state.copy_(x[:, -1])
        return tuple(x.clone() for _ in mixes)

    def value_residual(value, first_value, gate_bias, gate_delta):
        del first_value, gate_bias, gate_delta
        calls.append("value_residual")
        return value

    def key_gate(key, key_scale, gate_bias, gate_delta, key_gate_scale):
        del key_scale, gate_bias, gate_delta, key_gate_scale
        calls.append("key_gate")
        return key, -torch.ones_like(key), torch.full_like(key, 0.25)

    def output_transform(output, receptance, key, value, residual_scale, norm_weight, norm_bias, gate):
        del receptance, key, value, residual_scale, norm_weight, norm_bias, gate
        calls.append("output_transform")
        return output

    def recurrence_call(mode, **kwargs):
        recurrence.update(mode=mode, **kwargs)
        return torch.zeros_like(kwargs["r"]), None

    monkeypatch.setattr(layer_module, "can_use_flash_rwkv_inference", lambda *args, **kwargs: True)
    monkeypatch.setattr(layer_module.flash_rwkv, "infer_tmix_mix6_fp16", mix6)
    monkeypatch.setattr(layer_module.flash_rwkv, "infer_tmix_vres_gate_fp16", value_residual)
    monkeypatch.setattr(layer_module.flash_rwkv, "infer_tmix_kk_a_gate_fp16", key_gate)
    monkeypatch.setattr(layer_module.flash_rwkv, "infer_tmix_lnx_rkvres_xg_fp16", output_transform)
    monkeypatch.setattr(layer_module, "_run_rwkv7_operator", recurrence_call)

    layer = layer_module.RWKV7Attention(
        hidden_size=64,
        head_dim=64,
        layer_idx=1,
        value_dim=64,
        num_hidden_layers=2,
        fuse_norm=True,
    )
    hidden = torch.randn(1, 2, 64)
    first_value = torch.randn_like(hidden)
    with torch.no_grad():
        layer(hidden, v_first=first_value)

    assert calls == ["mix6", "value_residual", "key_gate", "output_transform"]
    assert recurrence["mode"] == "recurrent"
    assert recurrence["kk"] is None
    assert recurrence["a"] is None
    assert torch.equal(recurrence["flash_a"], -torch.ones_like(recurrence["flash_a"]))
    assert torch.equal(recurrence["flash_b"], torch.full_like(recurrence["flash_b"], 0.25))


def test_standard_feed_forward_inference_routes_flash_cmix(monkeypatch):
    import fla.models.rwkv7.modeling_rwkv7 as model_module

    calls = []

    def cmix(x, shift_state, mix):
        calls.append((x, shift_state, mix))
        shift_state.copy_(x[:, -1])
        return x

    monkeypatch.setattr(model_module, "can_use_flash_rwkv_inference", lambda *args, **kwargs: True)
    monkeypatch.setattr(model_module.flash_rwkv, "infer_cmix_mix_fp16", cmix)
    feed_forward = model_module.RWKV7FeedForward(
        hidden_size=64,
        intermediate_size=256,
        layer_idx=0,
        num_hidden_layers=1,
    )
    hidden = torch.randn(1, 2, 64)
    with torch.no_grad():
        output, state = feed_forward(hidden)

    assert output.shape == hidden.shape
    assert state is None
    assert len(calls) == 1
    assert calls[0][0] is hidden
    assert torch.equal(calls[0][1], hidden[:, -1])
    assert calls[0][2] is feed_forward.x_k


def test_flash_inference_eligibility_excludes_packed_and_training_paths(monkeypatch):
    from fla.ops.rwkv7.inference import can_use_flash_rwkv_inference

    class EligibleTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        @property
        def device(self):
            return torch.device("cuda")

        def is_contiguous(self):
            return True

    tensor = torch.Tensor._make_subclass(
        EligibleTensor,
        torch.empty(1, dtype=torch.float16),
        False,
    )
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: False)

    assert can_use_flash_rwkv_inference(tensor, head_dim=64)
    assert not can_use_flash_rwkv_inference(tensor, head_dim=32)
    assert not can_use_flash_rwkv_inference(tensor, cu_seqlens=object())

    monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    assert not can_use_flash_rwkv_inference(tensor, head_dim=64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_provider_standard_attention_inference_matches_unfused_path(monkeypatch):
    import fla.layers.rwkv7 as layer_module
    from fla.ops.rwkv7.inference import can_use_flash_rwkv_inference

    torch.manual_seed(2611)
    layer = layer_module.RWKV7Attention(
        hidden_size=128,
        head_dim=64,
        layer_idx=1,
        value_dim=128,
        num_hidden_layers=2,
        fuse_norm=True,
    ).cuda().half().eval()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.uniform_(-0.05, 0.05)
    hidden = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16).mul_(0.1)
    first_value = torch.randn_like(hidden).mul_(0.1)

    monkeypatch.setattr(layer_module, "can_use_flash_rwkv_inference", lambda *args, **kwargs: False)
    with torch.no_grad():
        expected = layer(hidden.clone(), v_first=first_value.clone())[0]

    calls = []
    for name in (
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
    ):
        operator = getattr(flash_api, name)

        def observe(*args, _name=name, _operator=operator, **kwargs):
            calls.append(_name)
            return _operator(*args, **kwargs)

        monkeypatch.setattr(flash_api, name, observe)
    monkeypatch.setattr(layer_module, "can_use_flash_rwkv_inference", can_use_flash_rwkv_inference)

    with torch.no_grad():
        actual = layer(hidden.clone(), v_first=first_value.clone())[0]

    assert calls == [
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
    ]
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert get_last_rwkv7_kernel() == "infer_tmix_lnx_rkvres_xg_fp16"
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_provider_standard_feed_forward_inference_matches_unfused_path(monkeypatch):
    import fla.models.rwkv7.modeling_rwkv7 as model_module
    from fla.ops.rwkv7.inference import can_use_flash_rwkv_inference

    torch.manual_seed(2612)
    feed_forward = model_module.RWKV7FeedForward(
        hidden_size=128,
        intermediate_size=256,
        layer_idx=0,
        num_hidden_layers=1,
    ).cuda().half().eval()
    with torch.no_grad():
        for parameter in feed_forward.parameters():
            parameter.uniform_(-0.05, 0.05)
    hidden = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16).mul_(0.1)

    monkeypatch.setattr(model_module, "can_use_flash_rwkv_inference", lambda *args, **kwargs: False)
    with torch.no_grad():
        expected = feed_forward(hidden.clone())[0]

    calls = []
    operator = flash_api.infer_cmix_mix_fp16

    def observe(*args, **kwargs):
        calls.append("infer_cmix_mix_fp16")
        return operator(*args, **kwargs)

    monkeypatch.setattr(flash_api, "infer_cmix_mix_fp16", observe)
    monkeypatch.setattr(model_module, "can_use_flash_rwkv_inference", can_use_flash_rwkv_inference)
    with torch.no_grad():
        actual = feed_forward(hidden.clone())[0]

    assert calls == ["infer_cmix_mix_fp16"]
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert get_last_rwkv7_kernel() == "infer_cmix_mix_fp16"
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
