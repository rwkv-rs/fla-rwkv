# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""RWKV7 backends."""

from fla.ops.backends import BackendRegistry, dispatch
from fla.ops.rwkv7.backends.flash_rwkv import FlashRWKVBackend

rwkv7_registry = BackendRegistry('rwkv7')
rwkv7_registry.register(FlashRWKVBackend())

__all__ = ['dispatch', 'rwkv7_registry']
