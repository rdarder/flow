"""JAX compilation cache configuration.

Provides setup function for persistent JAX compilation cache.
"""

import os
import pathlib

import jax


def setup_jax_compilation_cache(project_name: str = "barevision") -> None:
    """Configure JAX to use a persistent compilation cache.

    Cache location follows XDG spec:
    - Linux: ~/.cache/barevision/jax_cache
    - macOS: ~/Library/Caches/barevision/jax_cache
    - Or $XDG_CACHE_HOME/barevision/jax_cache if set

    Args:
        project_name: Name for cache directory (default: "barevision")
    """
    # Determine cache base directory
    cache_base = os.getenv("XDG_CACHE_HOME")
    if cache_base is None:
        cache_base = pathlib.Path.home() / ".cache"
    else:
        cache_base = pathlib.Path(cache_base)

    # Create project-specific cache path
    cache_path = cache_base / project_name / "jax_cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    # Configure JAX to use this cache
    jax.config.update("jax_compilation_cache_dir", str(cache_path))

    # Enable persistent cache (optional but recommended)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 1024)  # 1KB minimum


# Auto-setup when module is imported
setup_jax_compilation_cache()
