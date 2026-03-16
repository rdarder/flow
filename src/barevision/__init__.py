"""Barevision: Non-semantic perception for cheap robots.

This package contains modules for optical flow, embedding generation,
and other perception tasks targeting NPU-constrained hardware.
"""

# Configure JAX compilation cache (must be done before any JAX operations)
from barevision.utils.cache import setup_jax_compilation_cache

setup_jax_compilation_cache()

__version__ = "0.1.0"
