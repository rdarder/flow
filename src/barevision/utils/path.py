"""Utilities for path resolution and project configuration.

Provides consistent ways to locate project resources regardless of
current working directory.
"""

from pathlib import Path
from typing import Optional


def find_project_root(start_from: Optional[Path] = None) -> Path:
    """Find the project root by looking for pyproject.toml.

    Searches upward from the given path (or this module's location) until
    it finds a pyproject.toml file, which marks the project root.

    Args:
        start_from: Starting path for search. If None, uses the directory
                    containing this utils.py module (not cwd).

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If pyproject.toml not found in any parent directory
    """
    if start_from is None:
        # Start from this module's directory, not cwd
        # This ensures tests work from any working directory
        start_from = Path(__file__).resolve().parent.parent.parent
    elif not isinstance(start_from, Path):
        start_from = Path(start_from)

    for parent in [start_from, *start_from.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError(
        "Could not find project root (no pyproject.toml in path hierarchy). "
        f"Searched from: {start_from}"
    )


def get_datasets_dir() -> Path:
    """Get the path to the datasets directory.

    Returns:
        Path to datasets/frames directory

    Raises:
        RuntimeError: If datasets directory not found
    """
    project_root = find_project_root()
    datasets_dir = project_root / "datasets" / "frames"

    if not datasets_dir.exists():
        raise RuntimeError(
            f"Datasets directory not found: {datasets_dir}. "
            f"Project root is: {project_root}"
        )

    return datasets_dir


def get_checkpoints_dir() -> Path:
    """Get the path to the checkpoints directory.

    Returns:
        Path to checkpoints directory
    """
    project_root = find_project_root()
    checkpoints_dir = project_root / "checkpoints"

    # Create if doesn't exist
    checkpoints_dir.mkdir(exist_ok=True)

    return checkpoints_dir


def get_models_dir() -> Path:
    """Get the path to the models directory.

    Returns:
        Path to models directory
    """
    project_root = find_project_root()
    models_dir = project_root / "models"

    # Create if doesn't exist
    models_dir.mkdir(exist_ok=True)

    return models_dir
