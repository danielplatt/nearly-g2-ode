"""Compatibility shim for the round-S7 right-chart comparison diagnostic."""

from __future__ import annotations

import sys

from .s7 import right_chart_comparison as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    _impl.main()
