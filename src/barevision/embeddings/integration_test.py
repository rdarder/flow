"""Integration test: dataset + model + losses.

Demonstrates complete forward pass through the embedding pipeline.

Run:
    python -m barevision.embeddings.integration_test
"""

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import (
    self_attention_entropy_loss,
    cross_attention_entropy_loss,
    combined_loss,
)
from barevision.embeddings.video_dataset import VideoFrameDataset
from flax import nnx
import jax
import jax.numpy as jnp
import jax.random as jr


def test_dataset_batch():
    """Test loading and preprocessing a batch from dataset."""
    print("=" * 60)
    print("DATASET BATCH TEST")
    print("=" * 60)

    dataset = VideoFrameDataset(
        data_root="datasets/frames",
        split="train",
        max_frame_distance=3,
        img_size=(190, 190),
    )

    # Load a few samples
    batch_size = 4
    images1 = []
    images2 = []

    for i in range(batch_size):
        img1, img2, meta = dataset[i]
        images1.append(img1)
        images2.append(img2)

        print(f"  Sample {i}: {meta['video_name']} frame {meta['frame_t']}→{meta['frame_tk']} (dist={meta['distance']})")

    # Stack into batch
    batch1 = jnp.stack(images1)
    batch2 = jnp.stack(images2)

    print(f"\n  Batch 1 shape: {batch1.shape}")
    print(f"  Batch 2 shape: {batch2.shape}")
    print(f"  Value range: [{batch1.min():.3f}, {batch1.max():.3f}]")
    print("✓ Dataset batch loaded\n")

    return batch1, batch2


def test_model_forward(batch1, batch2):
    """Test model forward pass on batch."""
    print("=" * 60)
    print("MODEL FORWARD PASS")
    print("=" * 60)

    model = SimpleEmbeddingModel(
        embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
    )

    emb1 = model(batch1)
    emb2 = model(batch2)

    print(f"  Input: {batch1.shape}")
    print(f"  Embeddings 1: {emb1.shape}")
    print(f"  Embeddings 2: {emb2.shape}")
    print(f"  Embedding stats: mean={emb1.mean():.3f}, std={emb1.std():.3f}")
    print("✓ Model forward pass complete\n")

    return emb1, emb2, model


def test_losses(emb1, emb2):
    """Test loss computation on embeddings."""
    print("=" * 60)
    print("LOSS COMPUTATION")
    print("=" * 60)

    self_loss = self_attention_entropy_loss(emb1)
    cross_loss = cross_attention_entropy_loss(emb1, emb2)
    total = combined_loss(self_loss, cross_loss)

    print(f"  Self-attention loss: mean={self_loss.mean():.4f}, shape={self_loss.shape}")
    print(f"  Cross-attention loss: mean={cross_loss.mean():.4f}, shape={cross_loss.shape}")
    print(f"  Combined loss: mean={total.mean():.4f}, shape={total.shape}")
    print("✓ Losses computed\n")

    return self_loss, cross_loss, total


def test_gradient_flow(emb1, emb2):
    """Test gradient flow through complete pipeline."""
    print("=" * 60)
    print("GRADIENT FLOW")
    print("=" * 60)

    def total_loss_fn(e1, e2):
        self_l = self_attention_entropy_loss(e1).mean()
        cross_l = cross_attention_entropy_loss(e1, e2).mean()
        return self_l + cross_l

    grad1, grad2 = jax.grad(total_loss_fn, argnums=(0, 1))(emb1, emb2)

    print(f"  ∂loss/∂emb1: max={jnp.abs(grad1).max():.6f}")
    print(f"  ∂loss/∂emb2: max={jnp.abs(grad2).max():.6f}")

    assert jnp.abs(grad1).max() > 0, "Gradient for emb1 should be non-zero"
    assert jnp.abs(grad2).max() > 0, "Gradient for emb2 should be non-zero"
    print("✓ Gradients flow through entire pipeline\n")


def main():
    """Run integration test."""
    print("\n" + "=" * 60)
    print("EMBEDDING PIPELINE INTEGRATION TEST")
    print("=" * 60 + "\n")

    # Dataset → Model → Losses → Gradients
    batch1, batch2 = test_dataset_batch()
    emb1, emb2, model = test_model_forward(batch1, batch2)
    test_losses(emb1, emb2)
    test_gradient_flow(emb1, emb2)

    print("=" * 60)
    print("INTEGRATION TEST PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
