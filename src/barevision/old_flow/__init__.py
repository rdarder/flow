"""Barevision package initialization with JAX configuration."""

import os
from pathlib import Path

import jax
from platformdirs import user_cache_dir

# Determine cache directory
# Priority: BAREVISION_CACHE_DIR env var > user-specific cache directory
if os.environ.get("BAREVISION_CACHE_DIR"):
    CACHE_DIR = Path(os.environ["BAREVISION_CACHE_DIR"])
else:
    CACHE_DIR = Path(user_cache_dir("barevision"))

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configure JAX persistent compilation cache
jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR / "jax"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)
