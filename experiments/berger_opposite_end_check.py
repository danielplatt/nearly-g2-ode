"""Compatibility shim for the Berger opposite-end diagnostic."""

from __future__ import annotations

import sys

from .berger_space import opposite_end_check as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})

if __name__ != "__main__":
    sys.modules[__name__] = _impl
else:
    _impl.main()
