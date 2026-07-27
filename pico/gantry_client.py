#!/usr/bin/env python3
"""Compatibility launcher for the packaged OpenDerm Pico gantry client."""

from openderm.motion.pico.client import *  # noqa: F401,F403
from openderm.motion.pico.client import main


if __name__ == "__main__":
    main()
