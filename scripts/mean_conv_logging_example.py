"""Example: Integrating mean_conv_analysis into training loop.

This shows how to log mean convolution kernel diagnostics during training.
"""

# In your training.py, add the import:
from barevision.flow.mean_conv_analysis import log_mean_conv_analysis


# Then in your training loop, after computing gradients but before the optimizer step:
def train_step(model, batch, loss_fn, train_state):
    """Training step with mean_conv analysis."""

    # ... existing training code ...

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(model, batch)

    # ... optimizer step ...

    return loss, model


# In the main training loop where you log metrics:
for epoch in range(settings.training.epochs):
    for step, batch in enumerate(train_dataloader):
        global_step = epoch * len(train_dataloader) + step

        # Training step
        loss, model = train_step(model, batch, loss_fn, train_state)

        # Log training metrics
        if global_step % settings.logging.every_n_steps == 0:
            logger.scalar("train/loss", loss, step=global_step)

            # === Add mean_conv kernel analysis ===
            # This logs scalars every time (fast)
            model_state = nnx.state(model)
            log_mean_conv_analysis(
                logger=logger,
                model_state=model_state,
                num_levels=settings.model.num_levels,
                global_step=global_step,
                hidden_dim=settings.model.hidden_dim,
                log_histograms=True,  # Slower, maybe log less frequently
                log_images=True,  # Slowest, log even less frequently
                prefix="mean_conv_behavior",
            )

        # Optional: Log histograms and images less frequently
        if global_step % (settings.logging.every_n_steps * 10) == 0:
            model_state = nnx.state(model)
            log_mean_conv_analysis(
                logger=logger,
                model_state=model_state,
                num_levels=settings.model.num_levels,
                global_step=global_step,
                hidden_dim=settings.model.hidden_dim,
                log_histograms=True,
                log_images=True,
                prefix="mean_conv_behavior",
            )


# What you'll see in TensorBoard:
#
# mean_conv_behavior/level_0/
#   ├── weight_sum_mean          # Should stay near 1.0 (averaging behavior)
#   ├── weight_sum_std           # Low = all channels similar
#   ├── center_surround_ratio_mean  # >1 means low-pass, higher = more centered
#   ├── drift_from_init_mean     # How much kernels adapted from Gaussian init
#   ├── effective_sigma_mean     # Receptive field size (init: 1.0)
#   ├── channel_specialization   # High variance = channels specializing
#   └── positive_weight_ratio    # Near 1.0 = all positive (true averaging)
#
# mean_conv_behavior/level_0/histograms/
#   ├── weight_sums              # Distribution across 32 channels
#   ├── center_surround_ratios   # Shows if some channels are edge detectors
#   ├── drift_from_init          # Which channels adapted most
#   └── effective_sigma          # Distribution of receptive field sizes
#
# mean_conv_behavior/level_0/kernels_grid  # Image: 32 kernels as 8×4 grid


# Interpretation guide:
#
# ✓ Healthy training:
#   - weight_sum_mean ≈ 1.0 (maintaining averaging)
#   - center_surround_ratio > 1 (still low-pass)
#   - drift_from_init increases gradually (learning)
#   - effective_sigma stays in 0.5-2.0 range (reasonable RF)
#
# ⚠️ Warning signs:
#   - weight_sum_mean << 1 or >> 1 (not averaging anymore)
#   - center_surround_ratio < 1 (becoming edge detectors?)
#   - drift_from_init very high very fast (unstable training)
#   - effective_sigma > 3 (too spread out) or < 0.3 (too sharp)
#
# ℹ️ Interesting patterns:
#   - High channel_specialization = channels developing different roles
#   - Bimodal effective_sigma distribution = some sharp, some blurry kernels
#   - Negative weights appearing = learning more complex filters
