# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.rwkv6 import LoRA
from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import GroupNorm
from fla.modules.l2norm import l2_norm
from fla.modules.token_shift import token_shift
from fla.ops.rwkv7 import chunk_rwkv7, flash_rwkv, fused_mul_recurrent_rwkv7, recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from fla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from fla.ops.rwkv7.gate_output_correction import gate_output_correction
from fla.ops.rwkv7.inference import can_use_flash_rwkv_inference

if TYPE_CHECKING:
    from fla.models.utils import Cache


def _run_rwkv7_operator(
    mode: str,
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    decay_bias: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor | None,
    a: torch.Tensor | None,
    recurrent_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    flash_a: torch.Tensor | None = None,
    flash_b: torch.Tensor | None = None,
):
    if mode == 'recurrent':
        if (flash_a is None) != (flash_b is None):
            raise ValueError("FlashRWKV recurrence operands must be supplied together")
        if flash_a is None:
            if kk is None or a is None:
                raise ValueError("recurrent mode requires either FLA or FlashRWKV operands")
            recurrence_a = -kk
            recurrence_b = kk * a
        else:
            recurrence_a = flash_a
            recurrence_b = flash_b
        return recurrent_rwkv7(
            r=r,
            decay_logits=decay_logits,
            decay_bias=decay_bias,
            k=k,
            v=v,
            a=recurrence_a,
            b=recurrence_b,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    if mode == 'chunk':
        if kk is None or a is None:
            raise ValueError("chunk mode requires explicit normalized key and gate tensors")
        return chunk_rwkv7(
            r=r,
            decay_logits=decay_logits,
            decay_bias=decay_bias,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            safe_gate=True,
            chunk_size=64,
        )
    if mode == 'fused_recurrent':
        if kk is None or a is None:
            raise ValueError("fused_recurrent mode requires explicit normalized key and gate tensors")
        return fused_mul_recurrent_rwkv7(
            r=r,
            decay_logits=decay_logits,
            decay_bias=decay_bias,
            k=k,
            v=v,
            kk=kk,
            a=a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    raise ValueError(f"Not supported mode `{mode}`.")


def _lora_delta_and_bias(
    module: LoRA,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = module.lora[1](module.lora[0](x))
    projection = module.lora[2]
    if projection.bias is None:
        raise RuntimeError("FlashRWKV fused gates require an explicit LoRA bias")
    return F.linear(hidden, projection.weight), projection.bias


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'recurrent',
        hidden_size: int = 1024,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int | None = None,
        gate_low_rank_dim: int | None = None,
        a_low_rank_dim: int | None = None,
        v_low_rank_dim: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        num_hidden_layers: int = None,
        **kwargs,
    ) -> RWKV7Attention:
        super().__init__()

        if mode not in ['recurrent', 'chunk', 'fused_recurrent']:
            raise ValueError(f"Not supported mode `{mode}`.")
        self.mode = mode
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        # Increase lora dimension for headdim>64
        factor = self.head_dim / 64
        if decay_low_rank_dim is None:
            decay_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.decay_low_rank_dim = decay_low_rank_dim
        else:
            self.decay_low_rank_dim = decay_low_rank_dim

        if gate_low_rank_dim is None:
            gate_low_rank_dim = max(32, int(round((5 * (hidden_size**0.5)) / 32) * 32))
            self.gate_low_rank_dim = gate_low_rank_dim
        else:
            self.gate_low_rank_dim = gate_low_rank_dim

        if a_low_rank_dim is None:
            a_low_rank_dim = max(32, int(round((2.5 * (hidden_size**0.5)) * factor / 32) * 32))
            self.a_low_rank_dim = a_low_rank_dim
        else:
            self.a_low_rank_dim = a_low_rank_dim

        if v_low_rank_dim is None:
            v_low_rank_dim = max(32, int(round((1.7 * (hidden_size**0.5)) * factor / 32) * 32))
            self.v_low_rank_dim = v_low_rank_dim
        else:
            self.v_low_rank_dim = v_low_rank_dim

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm
        self.norm_eps = norm_eps

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=elementwise_affine,
                eps=self.head_dim*norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim*norm_eps,
                affine=elementwise_affine,
            )

        try:
            from transformers.modeling_utils import _init_weights
        except ImportError:
            _init_weights = True
        if _init_weights:
            self.apply(self._initialize_weights)
        for name, module in self.named_modules():
            module._in_rwkv_module = True

        warnings.warn(
            "According to Bo, you are using a potentially buggy FLA implementation of RWKV. "
            "If you plan to report any numbers based on this implementation, we strongly recommend "
            "cross-checking with the official repo: https://github.com/BlinkDL/RWKV-LM. "
            "Bo may disagree with results reported from this version.",
        )

    @torch.no_grad()
    @torch.compiler.disable
    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize only when we're processing the RWKV7Attention module itself
        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)  # 1 to ~0

            # Create position-based initialization tensor
            ddd = torch.ones(1, 1, self.hidden_size, device=self.x_r.device)
            www = torch.zeros(self.hidden_size, device=self.x_r.device)
            zigzag = torch.zeros(self.hidden_size, device=self.x_r.device)
            linear = torch.zeros(self.hidden_size, device=self.x_r.device)
            for n in range(self.hidden_size):
                linear[n] = n / (self.hidden_size-1) - 0.5
                zigzag[n] = ((n % self.head_dim) - ((self.head_dim-1) / 2)) / ((self.head_dim-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (self.hidden_size - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
                ddd[0, 0, n] = n / self.hidden_size

            # Initialize x_* parameters directly
            self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
            self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
            self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_k.dtype)
            self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).to(self.x_v.dtype)
            self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
            self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)

            # Initialize k_k, k_a, r_k
            nn.init.constant_(self.k_a, 1.02)
            nn.init.constant_(self.r_k, -0.04)
            self.k_k.data.copy_((torch.zeros(self.hidden_size, device=self.k_k.device) +
                                0.71 - linear*0.1).to(self.k_k.dtype))
            # Set specific bias values for LoRA modules
            # 0.5 comes from F.softplus
            self.w_lora.set_bias_value(www + 0.5 + zigzag*2.5)
            self.a_lora.set_bias_value(-0.19 + zigzag*0.3 + linear*0.4)

            # v0 initialization - ones (for non-first layers)
            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(0.73 - linear*0.4)

            # Initialize GroupNorm
            self.g_norm.weight.data[:] = ((self.layer_idx + 1) / self.num_hidden_layers) ** 0.7

            # Initialize Linear projections
            self._orthogonal_init(self.r_proj.weight)
            self._orthogonal_init(self.k_proj.weight, gain=0.1)
            self._orthogonal_init(self.v_proj.weight)
            self.o_proj.weight.data.zero_()

            # Clean up temporary tensors to free memory
            del ddd, www, zigzag, linear

        module._is_hf_initialized = True

    @staticmethod
    def _orthogonal_init(weight, gain=1.0):
        oringinal_dtype = weight.dtype
        weight = weight.float()
        nn.init.orthogonal_(weight, gain=gain)
        weight = weight.to(oringinal_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        v_first: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
            am = attention_mask.narrow(1, attention_mask.size(1) - seq_len, seq_len).unsqueeze(-1)

        last_state = get_layer_cache(self, past_key_values)

        if attention_mask is not None:
            hidden_states = hidden_states.mul(am)

        # delta [batch_size, seq_len, hidden_size]
        # conv_cache [N, D]
        if last_state is None:
            conv_cache = None
            recurrent_state = None
        else:
            conv_cache = last_state['conv_state']
            recurrent_state = last_state['recurrent_state']

        mixes = tuple(
            parameter.reshape(-1)
            for parameter in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        )
        flash_shift_state = (
            torch.zeros(
                batch_size,
                self.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if conv_cache is None
            else conv_cache
        )
        if can_use_flash_rwkv_inference(
            hidden_states,
            flash_shift_state,
            *mixes,
            cu_seqlens=cu_seqlens,
        ):
            xr, xw, xk, xv, xa, xg = flash_rwkv.infer_tmix_mix6_fp16(
                hidden_states,
                flash_shift_state,
                mixes,
            )
            conv_state = flash_shift_state
        else:
            delta, conv_state = token_shift(
                hidden_states, cu_seqlens, output_cache=True, cache=conv_cache,
            )
            xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(
                hidden_states,
                delta,
                self.x_r,
                self.x_w,
                self.x_k,
                self.x_v,
                self.x_a,
                self.x_g,
            )

        r = self.r_proj(xr)
        if self.mode == 'recurrent' and torch.is_grad_enabled():
            # FlashRWKV's native training autograd consumes combined raw z and
            # propagates dz through the fused retention transform.
            decay_logits = self.w_lora(xw)
            decay_bias = None
        else:
            # Inference and the local chunk kernel fuse this broadcast bias at
            # the consumer boundary, avoiding an add materialization.
            decay_logits, decay_bias = _lora_delta_and_bias(self.w_lora, xw)

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v_delta, v_bias = _lora_delta_and_bias(self.v_lora, xv)
            if can_use_flash_rwkv_inference(
                v,
                v_first,
                v_bias,
                v_delta,
                cu_seqlens=cu_seqlens,
            ):
                v = flash_rwkv.infer_tmix_vres_gate_fp16(
                    v,
                    v_first,
                    v_bias,
                    v_delta,
                )
            else:
                v = torch.lerp(v, v_first, torch.sigmoid(v_bias + v_delta))
        g = self.g_lora(xg)

        a_delta, a_bias = _lora_delta_and_bias(self.a_lora, xa)
        if self.mode == 'recurrent' and can_use_flash_rwkv_inference(
            k,
            self.k_k,
            a_bias,
            a_delta,
            self.k_a,
            cu_seqlens=cu_seqlens,
            head_dim=self.head_dim,
        ):
            k, flash_a, flash_b = flash_rwkv.infer_tmix_kk_a_gate_fp16(
                k,
                self.k_k,
                a_bias,
                a_delta,
                self.k_a,
            )
            kk = None
            a = None
        else:
            a = torch.sigmoid(a_bias + a_delta)
            if self.fuse_norm:
                kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
            else:
                kk = F.normalize(
                    rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim),
                    dim=-1,
                    p=2.0,
                )

            # Preserve the BF16-stable fused multiply-add path outside eligible FP16 inference.
            k = fused_k_rwkv7(k, a, self.k_a)
            flash_a = None
            flash_b = None

        # dealing with left-padding
        if attention_mask is not None:
            v = v * am

        r, decay_logits, k = map(
            lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim),
            (r, decay_logits, k),
        )
        if decay_bias is not None:
            decay_bias = decay_bias.reshape(self.num_heads, self.head_dim)
        if a is not None:
            a = rearrange(a, 'b t (h d) -> b t h d', d=self.head_dim)
        if flash_a is not None:
            flash_a = rearrange(flash_a, 'b t (h d) -> b t h d', d=self.head_dim)
            flash_b = rearrange(flash_b, 'b t (h d) -> b t h d', d=self.head_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        o, recurrent_state = _run_rwkv7_operator(
            self.mode,
            r=r,
            decay_logits=decay_logits,
            decay_bias=decay_bias,
            k=k,
            v=v,
            kk=kk,
            a=a,
            flash_a=flash_a,
            flash_b=flash_b,
            recurrent_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            offset=r.shape[1],
        )

        flat_o = rearrange(o, 'b t h d -> b t (h d)')
        flat_r = rearrange(r, 'b t h d -> b t (h d)')
        flat_k = rearrange(k, 'b t h d -> b t (h d)')
        flat_v = rearrange(v, 'b t h d -> b t (h d)')
        residual_scale = self.r_k.reshape(-1)
        norm_weight = self.g_norm.weight
        norm_bias = self.g_norm.bias
        if (
            self.mode == 'recurrent'
            and self.key_dim == self.value_dim
            and norm_weight is not None
            and norm_bias is not None
            and self.g_norm.eps == 64e-5
            and can_use_flash_rwkv_inference(
                flat_o,
                flat_r,
                flat_k,
                flat_v,
                residual_scale,
                norm_weight,
                norm_bias,
                g,
                cu_seqlens=cu_seqlens,
                head_dim=self.head_dim,
            )
        ):
            o = flash_rwkv.infer_tmix_lnx_rkvres_xg_fp16(
                flat_o,
                flat_r,
                flat_k,
                flat_v,
                residual_scale,
                norm_weight,
                norm_bias,
                g,
            )
        else:
            if self.fuse_norm:
                o = self.g_norm(flat_o)
            else:
                o = self.g_norm(flat_o.view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
            o = gate_output_correction(o, r, k, self.r_k, v, g)
        o = self.o_proj(o)

        return o, None, past_key_values, v_first
