# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Small native packed recurrent workload for Compute Sanitizer racecheck."""

import json

import torch

from fla.ops.rwkv7 import (
    FLASH_RWKV_SOURCE_REVISION,
    get_last_rwkv7_provider,
    recurrent_rwkv7,
    validate_flash_rwkv_installation,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    provenance = validate_flash_rwkv_installation()
    torch.manual_seed(20260802)
    shape = (1, 3, 1, 64)
    inputs = tuple(
        (torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02).contiguous()
        for _ in range(6)
    )
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    operations = []
    for mode, state_dtype in (
        ("fp32io16", torch.float32),
        ("fp16", torch.float16),
    ):
        state_pool = torch.zeros(6, 1, 64, 64, device="cuda", dtype=state_dtype)
        untouched_before = state_pool[[0, 2, 3, 5]].clone()
        output, final_state = recurrent_rwkv7(
            *inputs,
            initial_state=state_pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
        )
        torch.cuda.synchronize()
        if final_state is not state_pool:
            raise RuntimeError("packed final state lost input state-pool identity")
        if get_last_rwkv7_provider() != "flash_rwkv":
            raise RuntimeError("packed racecheck did not use FlashRWKV")
        if not torch.equal(state_pool[[0, 2, 3, 5]], untouched_before):
            raise RuntimeError("packed racecheck modified an unselected state row")
        if not torch.isfinite(output).all():
            raise RuntimeError("packed racecheck output is non-finite")
        operations.append(f"recurrent_rwkv7_packed_stateful_{mode}")
    print(
        json.dumps(
            {
                "operations": operations,
                "operator_count": len(operations),
                "provider": "flash_rwkv",
                "provider_revision": provenance.revision,
                "expected_provider_revision": FLASH_RWKV_SOURCE_REVISION,
                "device": torch.cuda.get_device_name(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
