import torch
import torch.nn as nn
import math


class SinusoidalPosEmb2D(nn.Module):
    def __init__(self, embed_dim, grid_size):
        """
        Args:
            embed_dim (int): The total embedding dimension (C).
            grid_size (int): The size of one side of the (square) grid (H or W).
        """
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by 4.")

        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # We'll create embeddings for (grid_size, grid_size, embed_dim)
        # We split embed_dim into two halves (embed_dim // 2)
        # One half for 'x' (width), one for 'y' (height)
        # Each half is further split for sin/cos

        half_dim = embed_dim // 2

        # Scale for the frequencies
        # 10000^(2i / half_dim)
        div_term = torch.exp(torch.arange(0., half_dim, 2) * -(math.log(10000.0) / half_dim))

        # Position grids
        pos_w = torch.arange(0., grid_size).unsqueeze(1)
        pos_h = torch.arange(0., grid_size).unsqueeze(1)

        # Create sin/cos embeddings for width (x)
        pos_emb_w = torch.zeros(grid_size, half_dim)
        pos_emb_w[:, 0::2] = torch.sin(pos_w * div_term)
        pos_emb_w[:, 1::2] = torch.cos(pos_w * div_term)

        # Create sin/cos embeddings for height (y)
        pos_emb_h = torch.zeros(grid_size, half_dim)
        pos_emb_h[:, 0::2] = torch.sin(pos_h * div_term)
        pos_emb_h[:, 1::2] = torch.cos(pos_h * div_term)

        # Combine them to (grid_size, grid_size, embed_dim)
        # [G, D/2] -> [G, 1, D/2] -> [G, G, D/2]
        pos_emb_w = pos_emb_w.unsqueeze(1).repeat(1, grid_size, 1)
        # [G, D/2] -> [1, G, D/2] -> [G, G, D/2]
        pos_emb_h = pos_emb_h.unsqueeze(0).repeat(grid_size, 1, 1)

        pos_emb = torch.cat([pos_emb_w, pos_emb_h], dim=-1)  # [G, G, D]

        # Register as a non-learnable buffer
        # Shape: [1, grid_size, grid_size, embed_dim]
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))

    def forward(self, x):
        """
        Input x: [B, H, W, C]
        Output: [B, H, W, C]
        """
        # x is [B, H, W, C], pos_emb is [1, H, W, C]
        # Broadcasting adds them together
        return x + self.pos_emb


class V0MicroFlow(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=16, n_heads=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        # This one Conv2D layer is our patch embedding
        self.patch_embed = nn.Sequential(
            # --- Block 1 (Input: 3, 32, 32) ---

            # 1. Depthwise (Spatial)
            nn.Conv2d(
                in_channels=3,
                out_channels=24,
                kernel_size=3,
                stride=2,  # -> [B, 3, 16, 16]
                padding=1,
                groups=3  # KEY: Makes it depthwise
            ),
            nn.GELU(),
            nn.LayerNorm([24, 16, 16]),

            # 2. Pointwise (Channel-mixing)
            nn.Conv2d(
                in_channels=24,
                out_channels=embed_dim // 2,
                kernel_size=1,
                stride=1  # -> [B, 32, 16, 16]
            ),
            nn.GELU(),
            nn.LayerNorm([embed_dim // 2, 16, 16]),

            # --- Block 2 (Input: 32, 16, 16) ---

            # 3. Depthwise (Spatial)
            nn.Conv2d(
                in_channels=embed_dim // 2,  # 32
                out_channels=embed_dim // 2,  # 32
                kernel_size=3,
                stride=2,  # -> [B, 32, 8, 8]
                padding=1,
                groups=embed_dim // 2  # 32 groups
            ),
            nn.GELU(),
            nn.LayerNorm([embed_dim // 2, 8, 8]),

            # 4. Pointwise (Channel-mixing)
            nn.Conv2d(
                in_channels=embed_dim // 2,  # 32
                out_channels=embed_dim,  # 64
                kernel_size=1,
                stride=1  # -> [B, 64, 8, 8]
            ),
            nn.GELU()
        )
        grid_size = img_size // patch_size  # 32 // 4 = 8
        self.grid_size = grid_size

        # 2. Positional Embedding
        self.pos_embed = SinusoidalPosEmb2D(embed_dim, grid_size)

        # 3. Cross-Attention
        # We only need one MultiheadAttention block
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True  # Expects [B, N, C]
        )

        # We need LayerNorms for stability
        self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)

        # 4. Decoder Head
        # A simple MLP to map from embed_dim -> 2 (dx, dy)
        # This is applied to each patch's token
        self.decoder_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2)
        )

    def _embed_patches(self, x):
        """Helper to turn (B, 3, H, W) -> (B, N, C)"""

        # x: [B, 3, 32, 32]
        # 1. Patch Embed: [B, C, H_grid, W_grid]
        x = self.patch_embed(x)  # [B, 64, 8, 8]

        # 2. Permute for positional embedding: [B, H_grid, W_grid, C]
        x = x.permute(0, 2, 3, 1)  # [B, 8, 8, 64]

        # 3. Add Positional Embedding
        x = self.pos_embed(x)  # [B, 8, 8, 64]

        # 4. Flatten for Transformer: [B, N, C]
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # [B, 64, 64]
        return x

    def forward(self, img1, img2):
        """
        img1 (Frame A): [B, 3, 32, 32]
        img2 (Frame B): [B, 3, 32, 32]
        """

        # Create patch embeddings for both images
        # tokens_a / tokens_b shape: [B, 64, 64] (B, N_patches, C_embed)
        tokens_a = self._embed_patches(img1)
        tokens_b = self._embed_patches(img2)

        # --- Core V0 Logic: Cross-Attention ---

        # Normalize tokens before attention
        q_norm = self.norm1(tokens_a)
        # K and V come from the same source (img2), so they share a norm
        # kv_norm = self.norm2(tokens_b)

        # Q = Frame A ("Where am I?")
        # K = Frame B ("Search here")
        # V = Frame B ("Get content from here")
        attn_output, _ = self.attn(
            query=tokens_a,
            key=tokens_b,
            value=tokens_b
        )

        # Add residual connection
        # This 'flow_features' tensor now contains information
        # about where each patch from A found its match in B.
        x = tokens_a + attn_output

        flow_features = self.norm1(x)

        # --- Decode Flow ---

        # Apply the MLP decoder to each patch's token
        # [B, 64, 64] -> [B, 64, 2]
        flow_pred = self.decoder_head(flow_features)

        # Reshape to our grid: [B, 8, 8, 2]
        flow_pred = flow_pred.reshape(
            -1, self.grid_size, self.grid_size, 2
        )

        # Pytorch convention is (B, C, H, W)
        # Let's make the output [B, 2, 8, 8]
        flow_pred = flow_pred.permute(0, 3, 1, 2)

        return flow_pred


# --- Test Script to Verify Shapes ---
if __name__ == "__main__":
    # 1. Create the model
    model = V0MicroFlow(
        img_size=32,
        patch_size=4,
        embed_dim=64,
        n_heads=4
    )

    # 2. Create dummy input data
    BATCH_SIZE = 4
    # img1 and img2 are our two frames
    dummy_img1 = torch.rand(BATCH_SIZE, 3, 32, 32)
    dummy_img2 = torch.rand(BATCH_SIZE, 3, 32, 32)

    # 3. Run the forward pass
    pred_flow = model(dummy_img1, dummy_img2)

    # 4. Check the output shape
    # We expect (B, 2, 8, 8)
    # B = 4
    # 2 = (dx, dy)
    # 8 = 32 / 4 (grid_size)
    print(f"Model created successfully.")
    print(f"Input image shape: {dummy_img1.shape}")
    print(f"Predicted flow shape: {pred_flow.shape}")

    expected_shape = (BATCH_SIZE, 2, 8, 8)
    assert pred_flow.shape == expected_shape

    print("\nShape test PASSED.")

    # Check number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
