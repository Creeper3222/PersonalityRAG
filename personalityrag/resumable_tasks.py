"""Compatibility alias for the LivingMemory v8 implementation."""

from __future__ import annotations

import sys

from .library_types.livingmemory_v8 import resumable_tasks as _implementation


sys.modules[__name__] = _implementation
