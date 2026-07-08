"""Compatibility shim for S7 maximal-volume calibration."""

from __future__ import annotations

from .s7.max_volume_calibration import main


if __name__ == "__main__":
    main()
