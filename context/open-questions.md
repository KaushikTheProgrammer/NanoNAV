# Open Questions

## Immediate (Pre-Training)

### Dataset size sufficiency
50 episodes at 30-60s = ~13K-18K transitions. The 43K target from the original doc may be needed. Train first checkpoint on initial collection, check diagnostic, scale from there. Upper bound for single room: ~4-6 hours (~130-190K transitions).

### Frame interval f tuning
f=5 (167ms chunks) matches PushT baseline. Navigation may benefit from larger f (more ground covered per model step, same CEM dimensionality). But larger f → larger Δx → coarser body-frame delta approximation. Test f=5 first, try f=8-10 if CEM reach is insufficient.

### Trajectory validation tool
Need to build offline visualization: take raw velocity logs, integrate body-frame deltas, plot world-frame trajectory, verify against odometry. Confirms the integration math before training.

**Update (2026-06-01):** The collected dataset (`wm-smallarea_merged`) has **no logged global pose** — `observation.state` mirrors the action (arm joints + base velocity), not pose. So there is no *independent* odometry to validate the SE(2) integration against; the world-frame trajectory we'd plot is itself derived from the same velocities. Validation is therefore **visual-flow consistency**: SD-VAE `compare` of frame *k* vs *k+f*, checking that flow direction/magnitude matches the sign/scale of the integrated `(Δx, Δθ)`. This is also exactly what training cares about. See [[nanowm-integration]].

### Forward-speed coverage (bang-bang data) — found 2026-06-01
Integration validation showed `x.vel` is near bang-bang: per-chunk Δx is bimodal (≈0 or ≈1.65 cm at
full speed), with few intermediate values. The slow/low-Δx regime needed for fine near-goal approach
is sparsely covered. Options if near-goal CEM struggles: collect a few deliberately-slow episodes
(as the original plan intended but the data under-delivered), or down-weight reliance on fine speed
control near the goal. See [[experiment-log]].

### Reach per step shorter than assumed — found 2026-06-01
Max Δx ≈ 1.65 cm/chunk (x.vel ≤ 0.1 m/s), so H=3 covers ~5 cm, not the design's ~15 cm. The
flat-scoring threshold (~30 cm) is hit after very few chunks → reinforces both the f=8–10 experiment
(below) and the waypoint scaffold ([[planning]]). Consider whether f=8–10 is needed from the start.

## Training Phase

### Will action branch survive with real-world data?
PushT action conditioning works in sim with clean renders. Real-world camera noise, slight lighting variations, and visual complexity might make unconditional prediction harder — which could either help (model NEEDS the action to predict) or hurt (too much visual noise drowns the action signal). Table 5/6 diagnostic is the gate.

### If Table 5/6 fails — fallback options
1. Add absolute global pose as auxiliary conditioning (environment-specific but maximally informative)
2. Try different action injection mechanism (cross-attention instead of additive — most expressive, most params)
3. Increase action embedding dimension
4. Data augmentation to force action sensitivity

## Planning Phase

### Waypoint graph construction details
- Spatial sampling interval (~30cm proposed — tune based on CEM scoring range)
- DepthAnything3 reconstruction quality on overhead camera frames
- Graph connectivity: metric threshold vs k-nearest-neighbors
- Localization: how to place current observation in the graph at runtime

### Waypoint switching
- Threshold on latent distance for advancing to next waypoint
- Timeout mechanism if MPC gets stuck
- Handling approach from unexpected direction (directionality problem)

### Scoring function alternatives
- Raw SD-VAE L2 (simplest, appearance-based)
- DINO feature distance for scoring only (semantic, heading-invariant) while predicting in SD-VAE (decoupled generation/scoring latents)
- Learned navigational distance predictor
- Pose-based scoring via DA3 localization in reconstruction

## Future Extensions

### Pattern B (goal-conditioned video generation + IDM)
Same data, same encoder. Replace action conditioning with goal-image conditioning. Train separate IDM on action-labeled data. Compare Pattern A vs Pattern B on same scene. Analytic optical-flow IDM baseline (AVDC-style, zero labels).

### Hierarchical World Model (HWM)
High-level CEM generates latent subgoals, low-level CEM plans actions. Eliminates external graph. Push-T success 17% → 61% with DINO-WM. More complex but more elegant.

### Mobile manipulation
Extend action space to include arm joints. Use both overhead and wrist cameras. The world model predicts visual consequences of both base motion and arm motion jointly.

### Multi-room / environment transfer
Current setup is single-room, single-lighting. Generalization requires diverse environments. Consider sim-to-real transfer via Isaac Sim LeKiwi nav environments.

### Latent actions (LAPA/CLAM)
Continuous latent action model reduces action-label requirement. Enables training on action-free video. Natural scaling path.

### Real-time planning
DDIM steps 20→5, CEM warm-starting, reduced samples. Target: replan within 167ms chunk duration for smooth continuous navigation.
