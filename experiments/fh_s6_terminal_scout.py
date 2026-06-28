"""Compatibility shim for FH S6 terminal-shooting scout grids."""

from __future__ import annotations

from .foscolo_haskins.s6_scout import main_terminal


if __name__ == "__main__":
    main_terminal()
