"""Compatibility shim for the S7 SU(2)^3 next-sprint audit."""

from __future__ import annotations

import sys

from .s7 import su2_cubed_next_sprint_audit as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})


if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    _impl.main()
