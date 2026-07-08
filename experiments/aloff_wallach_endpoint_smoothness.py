"""Compatibility shim for Aloff-Wallach N_{1,1} endpoint smoothness."""

from experiments.aloff_wallach.endpoint_smoothness import *  # noqa: F401,F403
from experiments.aloff_wallach.endpoint_smoothness import main


if __name__ == "__main__":
    main()
