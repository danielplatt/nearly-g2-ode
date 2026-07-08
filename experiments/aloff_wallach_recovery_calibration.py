"""Compatibility shim for the Aloff-Wallach known-solution recovery calibration."""

from experiments.aloff_wallach.recovery_calibration import *  # noqa: F401,F403
from experiments.aloff_wallach.recovery_calibration import main


if __name__ == "__main__":
    main()
