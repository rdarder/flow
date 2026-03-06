"""Test checkpoint manager for embeddings."""

import optax
from flax import nnx
import jax.random as jr

from barevision.embeddings.checkpoint_manager import create_checkpoint_manager
from barevision.embeddings.model import SimpleEmbeddingModel


def test_checkpoint_manager_smoke():
    """Test checkpoint save/restore."""
    model = SimpleEmbeddingModel(
        embed_dim=16,
        in_channels=3,
        rngs=nnx.Rngs(jr.PRNGKey(0)),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

    manager = create_checkpoint_manager(
        checkpoint_dir="test_checkpoints/test",
        save_interval_steps=5,
        max_to_keep=2,
        enabled=True,
    )

    # Test should_save
    assert not manager.should_save(0), "Should not save at step 0"
    assert manager.should_save(5), "Should save at step 5"
    assert manager.should_save(10), "Should save at step 10"
    assert not manager.should_save(6), "Should not save at step 6"

    # Test save
    manager.save(step=5, model=model, optimizer=optimizer, epoch=0)
    assert manager.latest_step() == 5, "Latest step should be 5"

    # Test restore
    model2 = SimpleEmbeddingModel(
        embed_dim=16,
        in_channels=3,
        rngs=nnx.Rngs(jr.PRNGKey(1)),  # Different seed
    )
    optimizer2 = nnx.Optimizer(model2, optax.adam(1e-4), wrt=nnx.Param)

    epoch, step = manager.restore(model2, optimizer2)
    assert epoch == 1, f"Expected epoch 1 (next after saved epoch 0), got {epoch}"
    assert step == 5, f"Expected step 5, got {step}"

    manager.close()

    # Cleanup
    import shutil

    shutil.rmtree("test_checkpoints/test", ignore_errors=True)

    print("✓ CheckpointManager smoke test passed")


if __name__ == "__main__":
    test_checkpoint_manager_smoke()
