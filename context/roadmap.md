# Roadmap

Staged execution plan with current status. This is the durable home for *execution* tracking;
[[experiment-log]] holds the design chronology, [[open-questions]] holds unresolved forward work,
and [[training-runs]] holds per-run training telemetry.

Legend: ✅ done · ▶️ in progress · ⬜ not started

---

## ✅ Stage 1 — Design

All architectural decisions settled and recorded in [[overview]]: Pattern A (action-conditioned
forward model + CEM/MPC), NanoWM-B/2, SD-VAE latent, body-frame `(Δx, Δθ)` action, elevated
third-person camera.

## ✅ Stage 2 — Data Collection

50 teleop episodes recorded and merged → `kaushikpraka/wm-smallarea_merged` (44,926 frames @ 30 Hz,
LeRobot v3.0, 9-D action = 6 arm joints + base `x.vel/y.vel/theta.vel`, cameras `front/wrist/top`).
See [[data-collection]].

## ▶️ Stage 3 — Dataset Build

Turn raw episodes into NanoWM-trainable samples. Two facts shape this stage (see
[[nanowm-integration]]): NanoWM *concatenates* per-step actions over `frame_interval` (so integration
must be added), and it pins `lerobot-datasets==2.1.0` while our data is v3.0.

- **3a — Derived dataset** `scripts/build_lekiwi_nav_dataset.py`: v3.0 → **30 Hz LeRobot v2.1**,
  `top` camera only, 2-D base-velocity action `[x.vel, theta.vel]` (raw, integration deferred to the
  dataloader). Output: `kaushikpraka/wm-smallarea_nav30`. **(script written; run pending)**
- **3b — Validation:** SD-VAE `compare` of frame *k* vs *k+f* — flow direction/magnitude vs
  `(Δx, Δθ)`. No independent odometry exists (state is velocity, not pose) → visual-flow consistency
  only. ⬜

## ⬜ Stage 4 — First Checkpoint

NanoWM-B/2, v-prediction, additive injection, SD-VAE. Integrated `(Δx, Δθ)` action via the
`integrate_se2` dataloader patch; `frame_interval=5` (the tunable reach knob). Trained on a single
**RunPod H100** (eff-bs 64, ~50K steps). Babysat by a pod-side agent per [[runpod-operator-guide]].

## ⬜ Stage 5 — Action-Conditioning Diagnostic (Table 5/6) — **critical gate**

`action_diagnostic.py`: roll out under GT / zero / random actions; GT latent-L2 must clearly beat
zero/random and action-embedding RMS must be ~0.1+. **Fail → fix training before any planning**
(aux pose, cross-attn injection, larger embed, augmentation). See [[training]].

## ⬜ Stage 6 — Short-Range Planner (CEM/MPC)

Stop-and-plan loop, CEM over the 6-D action space (H=3 × 2-D), latent-L2 scoring, decode-and-visualize
rollouts. Proves goal-reaching at <30 cm. Requires CEM action wiring for `integrate_se2`
(`planning_experiment.py`). See [[planning]].

## ⬜ Stage 7 — Long-Range Navigation

Topological waypoint graph (+ DepthAnything3 metric edges) is the recommended start; HWM / learned
distance as alternatives. Where most [[open-questions]] cluster. See [[planning]].

## ⬜ Stage 8 — Extensions (future)

Pattern B comparison, real-time planning, mobile manipulation (arm), multi-room transfer, latent
actions. See [[open-questions]].

---

## Current critical path

3a (run preprocessing) → 3b (validate) → fork+patch+configs → 4 (train on H100) → 5 (diagnostic gate).
