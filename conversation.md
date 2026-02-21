## Goal

Build an NPU-friendly algorithm that estimates dense depth maps and camera pose (6 DOF) from sequential dense optical flow fields (monocular video). The solution must run on cheap, restricted NPUs (e.g., Rockchip RV1109) without arbitrary memory access patterns like `gatherND` or image warping. Only standard convolutions, pooling, sliding windows, and element-wise math are allowed. The system must be self-supervised (no ground-truth depth) and handle independently moving objects.

## Instructions

- **Hard Constraint**: No `gatherND`, no per-pixel warping operations. Standard image/matrix warping is off the table. Use only convolutions, pooling, global reductions, and element-wise operations.
- **Selected Architecture**: A 4-stage sequential pipeline (non-iterative for now):
  1. **Moving Object Detection**: Generate a probabilistic or binary mask of static vs moving pixels using flow consensus/distribution analysis
  2. **Rotation Estimation**: Extract the rotational flow component (depth-independent, zero-divergence field) from masked flow
  3. **Translation Direction (FOE)**: Remove rotation from flow to isolate translation component; estimate Focus of Expansion (FOE) via voting/pooling
  4. **Depth Estimation**: Compute relative depth as `depth ∝ 1/|translation_flow_magnitude|` with learned or fixed scale
- **Scale Ambiguity**: Accept relative depth (consistent ratios across pixels), resolve absolute scale later or through other means.
- **Pose Representation**: Undetermined; choose based on what makes the math easiest (suggest: rotation angles + FOE position, or 6-vector).
- **Moving Object Handling**: Must filter moving objects BEFORE pose estimation to prevent contamination. Mask doesn't need to be tight; false positives (static pixels marked as moving) are acceptable, false negatives (moving pixels marked as static) should be minimized.
- **Initialization**: Start rotation from zero (training on isolated pairs); FOE and depth priors TBD per stage.
- **Confidence**: Defer confidence output implementation until basic pipeline works.

## Discoveries

- **Epipolar Decomposition**: Flow naturally decomposes into rotation (divergence-free, independent of depth) and translation (vectors point toward Focus of Expansion, magnitude depends on inverse depth). This allows sequential estimation without iterative warping.
- **Forward Rendering**: Computing flow from depth+pose uses only element-wise operations (back-project, transform, project) - no warping needed. This enables the "guess flow, compare to measured, update parameters" approach.
- **Moving Object Detection**: Can use flow direction consensus (circular mean) to identify outliers. Pixels with flow direction significantly different from the dominant direction are likely moving independently.
- **Joint Estimation Pitfall**: Attempting to estimate pose, depth, and object motion simultaneously creates mathematical incommensurability - depth is a state, pose/motion are deltas, and they interact multiplicatively rather than additively.
- **No Warping Temporal Continuity**: Cannot use warped depth maps from previous frames. Temporal consistency must be achieved through pose/velocity priors or feature-based state updates, not geometric warping.

## Accomplished

- **Exploration Phase**: Exhaustively explored and rejected joint optimization, iterative nudge-based refinement, and feature-tracking approaches due to NPU constraints or mathematical heterogeneity.
- **Architecture Selection**: Finalized on the 4-stage epipolar decomposition approach as the viable path.
- **Flow Model Context**: Reviewed existing flow estimation architecture (`BarebonesFlowModel`) which provides dense flow and confidence maps using hierarchical attention-based matching (16x16 windows, pyramid-based).
- **High-Level Design**: Defined the 4-stage pipeline structure and verified each stage can be implemented with NPU-friendly operations (convolutions, pooling, global statistics).
- **Next**: Ready to begin detailed specification of Stage 1 (Moving Object Detection), followed by Stage 2 (Rotation), Stage 3 (FOE), and Stage 4 (Depth).

## Relevant files / directories

- `/home/rdarder/dev/flow/src/model.py` - Contains `BarebonesFlowModel`, the flow estimation module that provides dense optical flow and confidence maps. Uses hierarchical pyramid with 16x16 window attention.
- `/home/rdarder/dev/flow/design.md` - Design document for the flow estimation architecture (V0/V1 design).
- `/home/rdarder/dev/flow/depth.md` - Project context document outlining NPU constraints, goals, and hardware
