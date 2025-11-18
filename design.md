### ## 1. 📝 The "V1-Plus" Design Doc: Hierarchical Barebones Flow

This is the full, multi-level algorithm we've designed, incorporating all our "V1-plus" ideas.

**Core Philosophy:** A hierarchical "predict-and-correct" model. We start with a coarse flow, get a `Flow_Hint`, and then use our "barebones" attention within a small window to *refine* that hint.

#### 🏛️ Architecture & Components

1.  **Feature Pyramid:** A single CNN stem (e.g., our depthwise-separable V0.5) is run on both `img1` and `img2`. We extract feature maps at multiple scales (e.g., L0=`8x8`, L1=`16x16`, L2=`32x32`). These are our `F` matrices.
2.  **Location Tensors (`L`):** Fixed, non-learnable tensors at each pyramid level (e.g., `L_L0` is `(64, 2)`, `L_L1` is `(256, 2)`) that store the `(x, y)` coordinate of each patch.
3.  **Flow Tensors:** A `Flow_L` tensor (e.g., `(8, 8, 2)`) is computed at each level.

#### 🔁 The Per-Level Refinement Algorithm

This algorithm runs at each level `L`, from coarsest to finest.

1.  **Get Flow Hint:**
    * **At L0 (Coarsest):** There is no hint. `Flow_Hint` is a `(B, P, 2)` tensor of all zeros. The "lookup window" is the *entire* L0 frame.
    * **At L1+ (Finer):** Take `Flow_L` from the previous level. Use `nn.Upsample` to scale it to the current level's dimensions. This is `Flow_Hint`.

2.  **Define Query & Window:**
    * **Query (`F1_query`):** The *entire* feature map for `img1` at this level. (Shape `(B, P_q, F_dim)`).
    * **Window (`F2_window`, `L_window`):** For each query patch, use its `Flow_Hint` to define a small `k x k` lookup window (e.g., `7x7`) in the `img2` feature map, centered at the *hinted* location. `F2_window` and `L_window` contain the features and *absolute* coordinates of the patches *only* within this window.

3.  **Sharpen Features (Ambiguity Dampening):**
    * Apply your "popular feature" dampening.
    * `popularity = F1_query.abs().mean(dim=1, keepdim=True)` (Find popular features)
    * `weights = 1.0 / (popularity + 1e-6)` (Get dampening weights)
    * `F1_sharp = F1_query * weights`
    * `F2_sharp = F2_window * weights` (Use the *same* weights)

4.  **Calculate Raw Similarity (`C_raw`):**
    * `C_raw = F1_sharp @ F2_sharp.T`
    * This is a `(B, P_q, P_w)` matrix (where `P_w` is the number of patches in the window, e.g., `7*7=49`).

5.  **Calculate Gated Hint Boost (`B_gated`):**
    * This is the "one-shot" broadcasted operation.
    * `loc_hint = L_query + Flow_Hint`
    * `diff_vectors = loc_hint.unsqueeze(2) - L_window.unsqueeze(1)`
    * `dist_sq = torch.sum(diff_vectors**2, dim=-1)`
    * `B_base = torch.exp(-dist_sq / (2 * sigma**2))` (Gaussian "bell curve")
    * `B_gated = 1.0 + (w * B_base)` (Your "centered-at-1" gated boost)

6.  **Combine & Softmax:**
    * `C_biased = C_raw * B_gated` (Apply the gated boost)
    * `C_norm = softmax(C_biased, dim=-1)` (The "soft argmax" selector)

7.  **Calculate Weighted Position (`A`):**
    * `A = C_norm @ L_window` (Shape `(B, P_q, 2)`)
    * This is the new "soft" absolute position of the best match.

8.  **Calculate Correction:**
    * `Flow_Correction = A - L_query`
    * This is the *new, local* flow vector (e.g., `(0.1, -0.2)`).

9.  **Get Final Flow for this Level:**
    * `Flow_L = Flow_Hint + Flow_Correction`
    * This final, corrected flow is used as the hint for the *next* level.

---

### ## 2. ✂️ The "V0 Chainsaw": A Minimal Testable Model

Our goal is to test the *single-most-important* new idea: **"Is `softmax(F1@F2.T) @ L` a learnable way to find flow?"**

We must "chainsaw" *everything* else from the V1 design.

* **CUT:** The entire **Hierarchy** (L0, L1, L2...). We'll use one level.
* **CUT:** **Flow Hints** (no hierarchy, no hints).
* **CUT:** **Lookup Windows** (no hint, so we must look *everywhere*).
* **CUT:** **Gated Boost** (no hint, so nothing to boost from).
* **CUT:** **Feature Sharpening** (a V1 optimization, not needed for V0).
* **CUT:** **Correction Blending** (no hint, so the flow we find *is* the final flow).

#### 🌱 The V0: "Barebones Global Flow"

Here is the minimal trainable loop. It re-uses our *proven* components.

1.  **DATA:** Use the `SyntheticFlowDataset` (V2, complex blobs). We *know* this data has a clean signal.
2.  **MODEL (Input: `img1`, `img2`):**
    * **Feature Stem:** Run *both* images through our **V0.5 depthwise-separable conv stem** to get `F1` and `F2`. (Shape: `(B, C, H, W)`, e.g., `(B, 64, 8, 8)`). We keep this because we *proved* it's the only stem that let gradients flow.
    * **Reshape:** Flatten the spatial dimensions: `F1` and `F2` become `(B, P, C)`, e.g., `(B, 64, 64)`.
    * **Location Tensor (`L`):** Create a *fixed, non-learnable* tensor `L` of shape `(P, 2)` (e.g., `(64, 2)`) that holds the `(x, y)` coordinates of each patch. This tensor is just "data" for the model.
    * **Correlation (`C_raw`):** `C_raw = F1 @ F2.T`. (Shape: `(B, 64, 64)`).
    * **Scale:** `C_scaled = C_raw / sqrt(C_dim)`. (Good practice, as we discussed).
    * **Softmax:** `C_norm = softmax(C_scaled, dim=-1)`.
    * **Weighted Position (`A`):** `A = C_norm @ L`. (Shape: `(B, 64, 2)`). This is the *entire* "attention" part of the model.
    * **Calculate Flow:** `Flow_pred = A - L.unsqueeze(0)`. (Broadcasting `L` to match the batch).
3.  **LOSS:**
    * `Flow_target` (from the dataset) must be reshaped to `(B, 64, 2)` to match `Flow_pred`.
    * `loss = L1Loss(Flow_pred, Flow_target)`.

This V0 is one clean, end-to-end model. It has *zero* "black box" attention modules. It tests *only* your "barebones" `(C@L)-L` idea. We can implement this in a single model file and see if the loss on the synthetic data plummets.
