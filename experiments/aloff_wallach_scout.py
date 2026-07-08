"""Compatibility shim for the Aloff-Wallach N_{1,1} scout runner."""

from experiments.aloff_wallach.scout import *  # noqa: F401,F403
from experiments.aloff_wallach.scout import main


if __name__ == "__main__":
    main()
