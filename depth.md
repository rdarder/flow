## Project Context: Monocular Depth and Pose from Optical Flow

**Goal:** I am trying to build an algorithm that extracts a dense depth map and camera pose estimation from sequential dense optical flow fields (monocular video).

**Strict Hardware Constraints (NPU-Bound):**
This algorithm must run on cheap, restricted NPUs.

* **The Dealbreaker:** I cannot use arbitrary memory access patterns like `gatherND`. Standard image/matrix "warping" is off the table.
* **The Requirement:** The solution must be built entirely on simple, NPU-friendly memory access patterns (e.g., standard convolutions, pooling, sliding windows, element-wise math).

## Overall Approach

Exploit epipolar geometry to decompose flow into separable components: first extract rotation (depth-independent), then find translation direction via Focus of Expansion, finally compute depth from translation magnitude. Filter out moving objects beforehand using flow consensus.

### Stages

- Stage 1: Moving Object Detection. Identify pixels that don't fit the rigid scene model using flow direction consensus. Generate a mask of "static" pixels.
- Stage 2: Rotation Estimation. Extract the rotational flow component from masked flow. Rotation creates characteristic curling patterns independent of scene depth.
- Stage 3: Translation Direction (FOE). Remove rotation from flow to isolate translation. Find the Focus of Expansion—the point all translation flow vectors point toward or away from.
- Stage 4: Depth Estimation. With known rotation and translation direction, depth is inversely proportional to translation flow magnitude. Compute relative depth up to scale.
