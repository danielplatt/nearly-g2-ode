"""Compatibility shim for the Berger wide surrogate search."""

from __future__ import annotations

import sys

from .berger_space import non_mirrored_surrogate_wide_search as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    _impl.main()
