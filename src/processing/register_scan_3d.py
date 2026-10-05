#!/usr/bin/env python3
"""Register a skin-scan capture folder in full 3D.

The public module owns the CLI lifecycle and delegates the four reconstruction
stages to :mod:`processing._reconstruction`. Keeping the orchestration here
preserves the command entry point and lets tests replace individual stages
without importing their implementation details.
"""

from __future__ import annotations

import sys

from ._reconstruction.exporter import _write_registration_outputs
from ._reconstruction.parser import _parse_args
from ._reconstruction.problem import _prepare_problem
from ._reconstruction.solver import _solve_registration
from .registration_features import _self_limit_memory
from .registration_geometry import project

__all__ = ["main", "project"]


def main():
    # The box is shared: a runaway allocation must kill this process cleanly,
    # never invoke the kernel OOM killer on other users.
    cap = _self_limit_memory()
    try:
        return _main(cap)
    except MemoryError:
        sys.exit(
            "openderm-register: aborted on MemoryError (self-imposed RAM "
            "cap; the kernel OOM killer was NOT invoked). Lower "
            "--texture-ppmm / raise --downscale, or free memory and rerun."
        )


def _main(mem_cap=None):
    args = _parse_args()
    problem = _prepare_problem(args, mem_cap)
    solution = _solve_registration(args, problem)
    _write_registration_outputs(args, problem, solution)


if __name__ == "__main__":
    main()
