# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import threading

_PROVIDER_STATE = threading.local()


def set_last_rwkv7_provider(provider: str) -> None:
    _PROVIDER_STATE.value = provider


def get_last_rwkv7_provider() -> str | None:
    """Return the provider selected by the last RWKV7 call in this thread."""
    return getattr(_PROVIDER_STATE, 'value', None)


__all__ = ['get_last_rwkv7_provider']
