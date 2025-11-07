import torch
from torchinfo import summary

# --- 1. Import your model ---
# (Make sure model.py is in the same directory)
from model import V0MicroFlow 

# --- 2. Define Hyperparameters ---
# (These must match the model you want to test)
IMG_SIZE = 32
PATCH_SIZE = 4
EMBED_DIM = 64
N_HEADS = 4

# --- 3. Instantiate the Model ---
# We're profiling our V0.5 (Depthwise Separable) model
model = V0MicroFlow(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    n_heads=N_HEADS
)

# --- 4. Define the Input Shapes ---
# We must tell torchinfo the exact size of our inputs
# (B, C, H, W)
BATCH_SIZE = 1 # We care about "per-sample" cost
img_input_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

# Our model takes two images, so we pass a list of inputs
input_data = [
    torch.randn(img_input_shape), 
    torch.randn(img_input_shape)
]

# --- 5. Run the Summary ---
print(f"--- Profiling Model: V0MicroFlow ---")
print(f"Params: embed_dim={EMBED_DIM}, n_heads={N_HEADS}\n")

# This is the magic command
# It gives us a layer-by-layer breakdown
model_summary = summary(
    model, 
    input_data=input_data,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    verbose=1 # 1 = show layer-by-layer
)

print("\n--- Summary ---")
print(f"Total Parameters: {model_summary.total_params:,}")
print(f"Total MACs (Mult-Adds): {model_summary.total_mult_adds:,}")
print(f"Estimated Total FLOPs (MACs * 2): {model_summary.total_mult_adds * 2:,}")
