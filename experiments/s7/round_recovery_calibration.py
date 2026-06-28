"""Calibrate recovery of the round-S7 fixed-chart solution."""

from __future__ import annotations

from .search_common import main_recovery


def main() -> None:
    """Run the round-S7 recovery calibration."""
    main_recovery("round")


if __name__ == "__main__":
    main()
