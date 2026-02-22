# Hierarchical Optical Flow: Design Document

## 1. What We're Trying to Build

We're building an optical flow estimator for a cheap NPU, and we have a serious constraint: we can't do complex memory operations like gatherND (warping) at inference time. This is going to be part of a larger SLAM pipeline, but for now, we're focusing just on flow estimation.

The key insight that unlocked this whole design was realizing we don't actually need to warp images to match them. Traditional optical flow methods warp frame 2 toward frame 1 using estimated flow, then compute residuals. But we can't do that efficiently on our NPU.

Instead, we turned to attention mechanisms. Attention can find correspondences between frames without ever moving pixels around. The problem is, attention doesn't scale to full images. A 1024×1024 attention matrix would be a million elements—way too big for our hardware.

So we need a way to use attention without blowing up the compute. The answer turned out to be hierarchical: build a pyramid, estimate flow at coarse resolution first, then refine at finer resolutions.

## 2. The Building Blocks We Already Have

Before we started thinking about hierarchies, we already had a working single-window model in `src/model.py`. Understanding how this works is crucial because the hierarchical system is essentially running this same logic at multiple scales.

### The Stem

The stem takes an 18×18 image and produces 16×16 feature embeddings. It uses depthwise separable convolutions: a 3×3 depthwise convolution reduces spatial dimensions by 2, then a 1×1 pointwise convolution projects to our embedding dimension. This is efficient and gives us just enough spatial context.

### PatchLookup

PatchLookup is our cross-attention layer. It's the heart of the system—it tries to match each embedding in frame 1 to embeddings in frame 2.

But here's the thing: pure visual similarity isn't enough. Imagine two patches that look exactly identical but are on opposite sides of the image. They shouldn't match. Even if two patches look the same, they're probably not the same pixel if they're far apart.

So PatchLookup uses two signals:

1. **Visual similarity**: Do these patches look alike? (computed via dot product)
2. **Spatial proximity**: Are they close to each other? (Gaussian kernel on normalized positions)

The spatial bias is crucial. It keeps the model sane by preventing matches across large distances. When we combine these scores and run softmax, we get attention weights that tell us where each pixel in frame 1 likely moved to in frame 2.

### PeerPropagation

PeerPropagation is self-attention within a single frame. It solves a real problem: some pixels just can't find good matches. Think about textureless regions or occluded areas. In these cases, PatchLookup returns low confidence and questionable flow.

PeerPropagation says: "If you can't find your match, look at your neighbors. If they look like you and have high confidence flow estimates, copy their flow."

We were genuinely surprised by how well this works. Without PeerPropagation, the model struggles on occlusions. With it, the flow estimates become much more coherent. The model essentially borrows flow information from confident neighbors to fill in the gaps.

## 3. The Problem with a Single Window

The current model processes a single 16×16 window. That means:

- Input images are tiny (18×18)
- Maximum detectable flow is limited by the window size
- We can't handle anything resembling real video resolutions

We need to scale this up. But we can't just make the window bigger because attention is O(N²). A 32×32 window would have a 1024×1024 attention matrix. That's already pushing our limits. Go to 64×64 and we're at 4096×4096—absolutely not happening.

So we need a different approach. We need to process large images in small chunks, but still capture large motions.

## 4. The Hierarchical Solution

The idea: build a pyramid. Estimate flow at coarse resolution first, then refine at finer resolutions.

Here's the intuition:

- **Coarse level**: Process a downsampled version of the image. Large motions appear as small motions here. We get a rough flow estimate that captures the "gist" of where things moved.
- **Fine level**: Process the full-resolution image, but use the coarse flow as a hint. The attention only needs to search in a small neighborhood because the coarse flow already got us close.
- **Blend**: Use confidence scores to decide when to trust the fine estimate versus fall back to coarse.

This lets us capture large motions (via the coarse level) while still maintaining fine detail (via the fine level), all while keeping our attention matrices at a manageable 16×16 size.

## 5. Making It Work: Key Design Decisions

### 5.1 How We Generate Embeddings (It's the Same at Every Level)

Here's the beautiful insight that unifies our entire pyramid: **every level does exactly the same thing.**

At each level, we take a 2×2 spatial region, flatten it into a vector, and pass it through a 1×1 convolution to produce a 16-dimensional embedding. That's it. Same operation, same code path, same logic everywhere.

The only thing that changes is what goes into that 2×2 region:

- **At the finest level (Level 1)**: The 2×2 region contains 4 pixels from the original image. For a grayscale image, that's 4 values; for RGB, that's 4×3 = 12 values. We flatten and project to 16 dimensions.

- **At coarser levels (Level 0)**: The 2×2 region contains 4 embeddings from the level below. Each embedding is 16-dimensional, so we have 4×16 = 64 values. We flatten and project to 16 dimensions.

Same pattern, same operation. The finest level just happens to input pixels instead of embeddings.

**Why this is elegant:** We don't need different code paths for different levels. We don't need special handling for "the bottom of the pyramid" versus "internal levels." It's all just: take 2×2, flatten, 1×1 conv → embedding.

**For v1 with 64×64 input:**

- **Level 1 (finest)**: Start with a 64×64 image. Group into 2×2 patches → you get a 32×32 grid where each position has 4 channels (the flattened 2×2 patch). Run a 1×1 conv (4→16) → 32×32×16 embeddings.

- **Level 0 (coarse)**: Take 2×2 blocks from Level 1's embeddings. You get a 16×16 grid where each position has 64 channels (4 embeddings × 16 dimensions each). Run a 1×1 conv (64→16) → 16×16×16 embeddings.

Both levels: 2×2 region → flatten → 1×1 conv → 16-dim embedding.

**Why 1×1 convolutions?** We use them because they don't change spatial dimensions. No padding headaches, no "did I accidentally get 15×15 instead of 16×16?" moments. The spatial arrangement is determined entirely by how we group things into 2×2 blocks, not by convolution kernel sizes. Clean, predictable, debuggable.

Each level learns its own 1×1 convolution weights because the semantics are different. Level 1 is learning to embed raw pixel patches. Level 0 is learning to compress and re-embed high-level features that have already been through attention. But the *structure* is identical—it's the same operation with different learned parameters.

This uniformity is what makes the architecture simple and extensible. Want to add Level 2? Same code. Want to go to Level -1 for even coarser motion? Same code. The pyramid just layers the same transformation on itself, like a fractal.

### 5.4 What about window boundaries?

We process 16×16 lookup windows independently. No overlap, no blending between windows. This is simple and keeps the architecture clean.

But there's a cost: a pixel at the window edge can't borrow flow from a pixel in the neighboring window during PeerPropagation. If you're on the left edge of your window, you can't look to your left neighbor even if that neighbor is in the adjacent window.

**Why we accept this**: For v1, simplicity wins. The hierarchical blending helps—if the fine level has low confidence at window edges (which it might, since edge pixels have fewer neighbors), we fall back to coarse flow. Also, PeerPropagation can still help within the window. We'll add overlapping windows in v2 if we need them.

### 5.5 How do we blend flow between levels?

This is absolutely crucial. We don't just upsample coarse flow and add it as a bias. That would force the fine level to learn relative corrections, which is awkward.

Instead, we do this:

1. Process fine level independently (no knowledge of coarse flow)
2. Upsample coarse flow (2× replication—each coarse pixel becomes a 2×2 block)
3. Per-pixel weighted blend based on confidence

The formula:
```
weight_fine = Conf_1
weight_coarse = 1 - Conf_1
Flow_final = (weight_fine * Flow_1 + weight_coarse * Flow_0_up) / (weight_fine + weight_coarse)
```

**The intuition**: If the fine level is confident (sharp attention peak), trust it completely. If it's uncertain (flat attention distribution), the coarse level probably captured the large-scale motion correctly, so use that instead.

This handles the "overflow" case automatically. If motion is so large that the fine level can't find matches within its 16×16 window, confidence will be low everywhere, and we'll mostly use coarse flow.

### 5.6 Why no warping?

Traditional optical flow methods warp frame 2 toward frame 1 using estimated flow, then compute residuals. This is elegant: if your flow estimate is perfect, the warped frame 2 should look identical to frame 1.

We can't do this at inference because warping requires gatherND—sampling pixels at non-integer coordinates based on the flow vectors. Our NPU doesn't support this efficiently.

**Our approach**: Pure attention-based matching. We never warp; we just find correspondences. This is less efficient for large motions (which is why we need the hierarchy), but it's completely NPU-friendly.

For training, we might add photometric loss (doing warping in the loss function), but the model itself never warps during the forward pass.

## 6. The Data Flow (Step by Step)

Let's walk through what happens during a forward pass with 64×64 input:

### Level 1 (Fine - 32×32 flow)

1. Take the 64×64 image pair
2. Patchify 2×2 → 32×32×4
3. 1×1 convolution → 32×32×16 embeddings
4. Grid into 4 windows (2×2 arrangement, each 16×16)
5. For each window:
   - **PatchLookup**: Match to corresponding window in frame 2 → flow estimate, confidence score
   - **PeerPropagation**: Refine using neighbors → better flow, updated confidence
6. Stitch the 4 windows together → Flow_1 (32×32), Conf_1 (32×32)

### Level 0 (Coarse - 16×16 flow)

1. Take Level 1's 32×32×16 embeddings
2. Reshape into 2×2 windows → 16×16×64
3. 1×1 convolution (64→16) → 16×16×16 embeddings
4. Single 16×16 window:
   - **PatchLookup** → Flow_0, Conf_0
   - **PeerPropagation** → refined Flow_0, Conf_0

### Blending

1. Upsample Flow_0 and Conf_0 by 2× → 32×32
2. Per-pixel weighted blend:
   - High Conf_1 → mostly Flow_1
   - Low Conf_1 → mostly upsampled Flow_0
3. Output: Flow_final (32×32)

## 7. What We Expect to Happen

### Training dynamics

We expect the levels to specialize:
- **Level 0** learns to capture large-scale motion—the "gist" of where things moved
- **Level 1** learns fine details and corrections
- **Confidence scores** should correlate with actual accuracy
- **PeerPropagation** should fill in gaps, especially at occlusion boundaries

### Edge cases

- **Large uniform motion**: Fine level will have low confidence (can't find matches within small windows), blend will favor coarse
- **Textureless regions**: PeerPropagation should copy flow from textured neighbors
- **Occlusions**: PeerPropagation should handle this by borrowing from visible regions

### Failure modes we're watching for

- **Block artifacts at window boundaries**: If PeerPropagation can't help across windows, we might see visible seams
- **Over-reliance on coarse flow**: If fine confidence is systematically low, we'll never get fine detail
- **Confidence miscalibration**: If the softmax temperature isn't right, confidence won't correlate with accuracy

## 8. Future Directions (v2+)

### Shift/Crop

Instead of just upsampling coarse flow, we could use the average flow to physically shift or crop frame 2 before matching. This recenters the search window without doing actual warping. It's more efficient than searching the full 16×16 window.

### Overlapping Windows

Process 18×18 regions but only keep the center 16×16. This gives PeerPropagation access to neighbors across window boundaries and should reduce block artifacts.

### Learned Upsampling

Instead of 2× replication for upsampling flow, use a small convolution to upsample smoothly. This might preserve edges better.

### Confidence Calibration

Add learned temperature per level so confidence scores are comparable across scales. Right now, the softmax temperature is fixed, which might mean coarse and fine confidences aren't on the same scale.

### More Levels

Add Level 2 (128×128 embeddings, 8×8 windows) or Level -1 (8×8 embeddings for even coarser motion). The architecture scales naturally.

## 9. Summary

We're building a hierarchical optical flow model that:
- Uses attention (not warping) for NPU compatibility
- Processes 16×16 lookup windows independently (tractable attention size)
- Builds an embedding pyramid (fine to coarse via learned downsampling)
- Blends flow estimates using confidence (automatic coarse-to-fine selection)
- Uses PeerPropagation to fill gaps (occlusions, textureless regions)

For v1: 2 levels, 64×64 images, 1×1 convolutions only, non-overlapping windows.

The goal is to prove the concept: can we estimate flow hierarchically without warping, using only attention and confidence-based blending?

If this works, we have a path to handling larger images, more levels, and eventually integrating this into the full SLAM pipeline.
