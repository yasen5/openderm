#!/usr/bin/env python3
"""Compatibility launcher for the packaged OpenDerm Pico gantry client."""

from capture.motion.pico.client import *  # noqa: F401,F403
from capture.motion.pico.client import main


if __name__ == "__main__":
    main()
