import shutil

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import json
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Dict, Any, Tuple


# --- 1. Model Definitions ---

class PositionalEncoding(nnx.Module):
    """
    Sinusoidal Positional Encoding for 2D coordinates.
    Takes (..., 2) cartesian coords and returns (..., n_dim) PE.
    """

    def __init__(self, n_dim: int, div_term_base: float = 10000.0, *, rngs: nnx.Rngs):
        self.n_dim = n_dim
        self.div_term_base = div_term_base

        # n_dim must be divisible by 4
        if n_dim % 4 != 0:
            raise ValueError(f"n_dim ({n_dim}) must be divisible by 4.")

        n_freq = n_dim // 4
        i = jnp.arange(n_freq, dtype=jnp.float32)

        # (1, n_freq)
        div_term = jnp.power(self.div_term_base, (2 * i) / (n_dim // 2))
        self.div_term = nnx.Variable(
            jnp.reshape(div_term, (1, -1)),
            metadata={'is_state': True}
        )

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        # xy shape: (..., 2)
        x = xy[..., 0:1]  # (..., 1)
        y = xy[..., 1:2]  # (..., 1)

        # div_term shape: (1, n_freq)
        div_term = self.div_term.value

        # Broadcast (..., 1) with (1, n_freq) -> (..., n_freq)
        pe_x_sin = jnp.sin(x / div_term)
        pe_x_cos = jnp.cos(x / div_term)
        pe_y_sin = jnp.sin(y / div_term)
        pe_y_cos = jnp.cos(y / div_term)

        # Concatenate: [sin(x), cos(x), sin(y), cos(y)]
        # Shape: (..., n_dim)
        pe = jnp.concatenate([pe_x_sin, pe_x_cos, pe_y_sin, pe_y_cos], axis=-1)
        return pe


class Decoder(nnx.Module):
    """ A simple MLP to decode PE -> (x, y) coordinates. """

    def __init__(self, pe_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.dense1 = nnx.Linear(pe_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dense3 = nnx.Linear(hidden_dim, 2, rngs=rngs)  # Output (x, y)

    def __call__(self, pe: jnp.ndarray) -> jnp.ndarray:
        x = self.dense1(pe)
        x = nnx.relu(x)
        x = self.dense2(x)
        x = nnx.relu(x)
        x = self.dense3(x)
        return x


# --- 2. JIT-compiled Helper Functions ---

def _get_batch(key: jax.Array, config: Dict[str, Any], pe_generator: PositionalEncoding):
    """ Generates a batch of (PE_input, Cartesian_GT) data. """
    coords_gt = jax.random.uniform(
        key,
        shape=(config['batch_size'], 2),
        minval=config['coord_min'],
        maxval=config['coord_max']
    )
    pe_input = pe_generator(coords_gt)
    return pe_input, coords_gt


@nnx.jit
def _loss_fn(model: Decoder, pe_batch: jax.Array, coords_gt_batch: jax.Array) -> jnp.ndarray:
    """ Calculates the L2 (MSE) loss. """
    coords_pred_batch = model(pe_batch)
    loss = jnp.mean(jnp.square(coords_pred_batch - coords_gt_batch))
    return loss


@nnx.jit
def train_step(
        model: Decoder,
        optimizer: nnx.Optimizer,
        pe_batch: jax.Array,
        coords_gt_batch: jax.Array,
) -> jnp.ndarray:
    """
    Performs a single training step.
    This function is JIT-compiled and updates the model state in place.
    """
    # nnx.value_and_grad computes gradients w.r.t. the model's nnx.Param state
    (loss, grads) = nnx.value_and_grad(_loss_fn)(model, pe_batch, coords_gt_batch)

    optimizer.update(model, grads)
    return loss


# --- 3. Public API Functions ---

def train_and_persist_model(config: Dict[str, Any], save_dir: Path) -> None:
    """
    Initializes and trains a new decoder model, saving the result.

    Args:
        config: A dictionary with all training and model hyperparameters.
        save_dir: The directory to save the config and model checkpoint.
    """
    print(f"--- Starting Decoder Training ---")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Saving to: {save_dir}")
    print("-" * 30)

    # --- Initialization ---
    key = jax.random.PRNGKey(config['seed'])
    key, pe_key, decoder_key, data_key = jax.random.split(key, 4)

    pe_generator = PositionalEncoding(
        n_dim=config['pe_dim'],
        div_term_base=config['div_term_base'],
        rngs=nnx.Rngs(0)
    )

    decoder = Decoder(
        pe_dim=config['pe_dim'],
        hidden_dim=config['hidden_dim'],
        rngs=nnx.Rngs(params=decoder_key)
    )
    optimizer = nnx.Optimizer(
        decoder,
        optax.adamw(learning_rate=config['learning_rate']),
        wrt=nnx.Param
    )

    # --- Training Loop ---
    start_time = time.time()
    for step in range(config['total_steps']):
        data_key, step_key = jax.random.split(data_key)
        pe_batch, coords_gt_batch = _get_batch(step_key, config, pe_generator)

        loss = train_step(
            decoder, optimizer, pe_batch, coords_gt_batch
        )

        if step % config['log_every'] == 0 or step == config['total_steps'] - 1:
            elapsed = time.time() - start_time
            print(f"  Step: {step:5d} | Loss (MSE): {loss:10.8f} | Time: {elapsed:.2f}s")

    print("-" * 30)
    print("Training complete.")

    # --- Save Artifacts ---
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save config
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        # Sort keys for consistent human-readable order
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Saved config to {config_path}")

    # 2. Save model state
    _, state = nnx.split(decoder)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = (save_dir / 'state').absolute()
    if checkpoint_path.exists():
        print(f"checkpoint path {checkpoint_path} already exists. removing.")
        shutil.rmtree(checkpoint_path)
    checkpointer.save(checkpoint_path, state)
    checkpointer.wait_until_finished()

    print(f"Saved model to {save_dir}/state")
    print("--- Training Finished ---")


def load_decoder_model(
        load_dir: Path
) -> Tuple[Decoder, PositionalEncoding, Dict[str, Any]]:
    """
    Loads a persisted decoder model and its config from a directory.

    Args:
        load_dir: The directory containing the config.json and checkpoint.

    Returns:
        A tuple of (restored_decoder_model, pe_generator, config)
    """
    print(f"\n--- Loading Decoder Model ---")
    print('not yet implemented')
    config_path = load_dir / "config.json"

    # 1. Load Config
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {load_dir}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config: {json.dumps(config, indent=2)}")

    # 2. Build "Shell" Models from Config
    pe_generator = PositionalEncoding(
        n_dim=config['pe_dim'],
        div_term_base=config['div_term_base'],
        rngs=nnx.Rngs(0)  # This key doesn't matter for state-less module
    )

    def make_decoder():
        return Decoder(
            pe_dim=config['pe_dim'],
            hidden_dim=config['hidden_dim'],
            rngs=nnx.Rngs(params=jax.random.PRNGKey(0))  # Key is a placeholder
        )

    abstract_model = nnx.eval_shape(make_decoder)
    graphdef, abstract_state = nnx.split(abstract_model)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = (load_dir / 'state').absolute()
    state_restored = checkpointer.restore(checkpoint_path, abstract_state)
    model = nnx.merge(graphdef, state_restored)

    print(f"Restored model from {load_dir}")
    print("--- Loading Finished ---")

    return model, pe_generator, config


def evaluate_model(
        decoder: Decoder,
        pe_generator: PositionalEncoding,
        config: Dict[str, Any],
        num_samples: int = 1_000_000
) -> float:
    """
    Runs a large-scale evaluation on a trained decoder model.

    Args:
        decoder: The trained Decoder model.
        pe_generator: The corresponding PE generator.
        config: The config dict (for batch_size, min/max coords).
        num_samples: Total number of samples to test.

    Returns:
        The final average MSE.
    """
    print(f"\n--- Running Evaluation ---")
    print(f"Evaluating on {num_samples:,} random samples...")

    key = jax.random.PRNGKey(config['seed'] + 1)  # Use a different seed
    batch_size = config['batch_size']
    num_batches = max(1, num_samples // batch_size)

    total_loss = 0.0

    # Create a JIT-compiled version of the loss function
    jitted_loss_fn = nnx.jit(_loss_fn)

    for i in range(num_batches):
        key, batch_key = jax.random.split(key)
        pe_batch, coords_gt_batch = _get_batch(batch_key, config, pe_generator)

        loss = jitted_loss_fn(decoder, pe_batch, coords_gt_batch)
        total_loss += loss

    avg_loss = total_loss / num_batches
    avg_pixel_error = jnp.sqrt(avg_loss)

    print(f"Evaluation complete.")
    print(f"  Final MSE: {avg_loss:10.8f}")
    print(f"  Avg. Pixel Error (RMSE): {avg_pixel_error:10.8f} pixels")
    print("--- Evaluation Finished ---")

    return float(avg_loss)


# --- 4. Main script execution ---

if __name__ == "__main__":
    # Define the default experiment
    DEFAULT_CONFIG = {
        'pe_dim': 16,
        'hidden_dim': 4,
        'div_term_base': 4000.0,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'total_steps': 30000,
        'log_every': 1000,
        'coord_min': -100.0,
        'coord_max': 100.0,
        'seed': 0,
    }

    DEFAULT_SAVE_DIR = Path("./pe_decoder_v1")

    # 1. Train and save the model
    train_and_persist_model(DEFAULT_CONFIG, DEFAULT_SAVE_DIR)

    # 2. Load the model back (to prove persistence works)
    decoder, pe_gen, config = load_decoder_model(DEFAULT_SAVE_DIR)
    #
    # 3. Run a final, large evaluation
    evaluate_model(decoder, pe_gen, config, num_samples=1_000_000)
