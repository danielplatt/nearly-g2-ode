"""Compatibility shim for the historical mirrored round-S7 probe."""

from __future__ import annotations

import sys

from .s7 import mirrored_probe as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    _impl.main()
