"""CLI wrapper for S7 Sp(1)xSp(1)xU(1) max-volume matching diagnostics."""

from __future__ import annotations

import argparse
import json

from .s7.sp1xsp1xu1_matching import known_recovery_summary


def main(argv: list[str] | None = None) -> int:
    """Run known round/squashed max-volume recovery diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-known", action="store_true", help="run known exact-germ recoveries")
    args = parser.parse_args(argv)
    if not args.recover_known:
        parser.error("only --recover-known is currently supported")
    print(json.dumps(known_recovery_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
