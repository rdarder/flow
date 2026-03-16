"""Barevision: Non-semantic perception for cheap robots.

This package contains modules for optical flow, embedding generation,
and other perception tasks targeting NPU-constrained hardware.
"""

# Configure JAX compilation cache (must be done before any JAX operations)
from barevision.utils import cache  # noqa: F401

__version__ = "0.1.0"
