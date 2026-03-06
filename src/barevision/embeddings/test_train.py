"""Smoke test for training script.

Run:
    pytest src/barevision/embeddings/test_train.py
"""

import subprocess
import sys


def test_smoke_train():
    """Test that training smoke test completes successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "barevision.embeddings.train", "--smoke-test"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Check successful completion
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Check expected output
    assert "TRAINING COMPLETE" in result.stdout
    assert "Model parameters: 328" in result.stdout
    assert "Epoch 0 complete" in result.stdout
    assert "Avg loss:" in result.stdout

    # Verify loss is finite (not NaN/Inf)
    for line in result.stdout.split("\n"):
        if "Loss:" in line:
            # Extract loss value
            parts = line.split("Loss:")
            if len(parts) > 1:
                loss_str = parts[1].strip().split()[0]
                loss = float(loss_str)
                assert loss > 0, f"Loss should be positive, got {loss}"
                assert loss < 10, f"Loss unexpectedly high: {loss}"


def test_model_initialization():
    """Test that model initializes with correct parameter count."""
    from barevision.embeddings.model import SimpleEmbeddingModel
    from flax import nnx
    import jax.random as jr

    model = SimpleEmbeddingModel(
        embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
    )

    # Count parameters (same logic as train.py)
    state = nnx.state(model)
    param_count = 0
    for module_state in state.values():
        for param_value in module_state.values():
            if hasattr(param_value, "size"):
                param_count += param_value.size

    assert param_count == 328, f"Expected 328 parameters, got {param_count}"
