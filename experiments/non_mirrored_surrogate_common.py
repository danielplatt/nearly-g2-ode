"""Compatibility shim for shared non-mirrored surrogate helpers."""

from __future__ import annotations

import sys

from .shared import non_mirrored_surrogate_common as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
sys.modules[__name__] = _impl
