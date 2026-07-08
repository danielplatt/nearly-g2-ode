"""Compatibility shim for the Aloff-Wallach homogeneous recovery check."""

from experiments.aloff_wallach.homogeneous_recovery import *  # noqa: F401,F403
from experiments.aloff_wallach.homogeneous_recovery import main


if __name__ == "__main__":
    main()
