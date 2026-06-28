"""Compatibility shim for shared mirror-sweep helpers."""

from __future__ import annotations

import sys

from .shared import mirror_sweep_common as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
sys.modules[__name__] = _impl
