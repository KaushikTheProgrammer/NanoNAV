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

## ✅ Stage 3 — Dataset Build

Turn raw episodes into NanoWM-trainable samples. Two facts shape this stage (see
[[nanowm-integration]]): NanoWM *concatenates* per-step actions over `frame_interval` (so integration
must be added), and the loadable LeRobot release for the v2.x format is **`lerobot==0.3.3`** (the
"v2.1" / "v3.0" in these notes is the dataset *codec* version, NOT a package version — the original
`lerobot-datasets==2.1.0` pin was a non-existent package; see [[nanowm-integration]]).

- **3a — Derived dataset** `scripts/build_lekiwi_nav_dataset.py`: v3.0 → **30 Hz LeRobot v2.1**,
  `top` camera only, 2-D base-velocity action `[x.vel, theta.vel]` (raw, integration deferred to the
  dataloader). **✅ Built to `/workspace/data/lekiwi` — 50 episodes / 44,926 frames; loads + decodes
  in NanoWM.** Built via a parallel decode-once → sharded-encode → merge pipeline (`--extract-frames`
  / `--frames-cache` / `--episode-slice` + `scripts/merge_lekiwi_shards.py`), ~6 min vs ~45–60 min
  sequential; output verified byte-identical to a sequential build.
- **3b — Validation:** load + decode + action-range sanity ✅ (vx∈[0,0.1] m/s, ω∈[−0.32,0.34] rad/s).
  SD-VAE `compare` of frame *k* vs *k+f* (visual-flow vs `(Δx, Δθ)`) still ⬜ — no independent
  odometry exists (state is velocity, not pose).

## ✅ Stage 4 — First Checkpoint (trained; overfit early — see Stage 5)

NanoWM-B/2, v-prediction, additive injection, SD-VAE. Integrated `(Δx, Δθ)` action via the
`integrate_se2` dataloader patch; `frame_interval=5` (the tunable reach knob). Trained on a single
**RunPod H100** (eff-bs 64, f=5). **Run 001** (wandb `x3ub`) trained on the uv-venv stack, but
**overfit by epoch ~3** (50 episodes is tiny for B/2; 50K steps = ~81 epochs, and the config saved no
best-val checkpoint) and was stopped at ~23K steps. See [[runpod-setup]], [[training-runs]].

## ❌ Stage 5 — Action-Conditioning Diagnostic (Table 5/6) — **FAILED on Run 001 (training cause, retry as Run 002)**

`action_diagnostic.py` (GT / zero / random rollouts): GT latent-L2 must clearly beat zero/random and
action-embedding RMS must be ~0.1+. **Run 001 (step-10K ckpt): FAIL** — RMS **0.0088**, GT 37.8 vs
zero 42.0 / random 42.4.

**Root cause CORRECTED (2026-06-03): a TRAINING problem, not observability.** A controlled
stationary-vs-translation latent contrast (`stationary_vs_translation.py`,
`viz/stationary-vs-translation/`) shows **translation IS observable** — pure-translation chunks change
the SD-VAE latent ~2× more than stationary ones (AUC 0.94 @ f=5 → 0.98 @ f=10), with a clean
dose-response and a near-field-floor parallax footprint. The earlier `corr(|Δx|, latentL2)≈0` (and the
2026-06-02 "translation unobservable / f refuted" conclusion) was an artifact of bang-bang Δx + pooled
rotation chunks. ⇒ The diagnosed checkpoint was simply **overfit** (step-10K = epoch 16; val bottomed
~epoch 3, no best-val ckpt kept) **at a low-SNR f=5** (translation ≈ noise floor). **Fix = Run 002**
(retrain at **f=10**, best-val checkpointing, low `max_steps`; re-diagnose the best-val model). Camera
relocation / odometry conditioning is demoted to a fallback. See [[training-runs]] (Run 002 plan),
[[open-questions]], [[experiment-log]].

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

✅ 3a (built) → ✅ 4 (Run 001 trained, overfit) → ❌ 5 (diagnostic FAILED on an **overfit f=5**
checkpoint — *not* an observability problem; translation is observable, see
`viz/stationary-vs-translation/`) → **▶️ Run 002: retrain at f=10 with best-val checkpointing + low
max_steps, then re-run the diagnostic on the best-val model (+ per-component Δx/Δθ sensitivity)** →
(gate must pass before) 6 (planner). The diagnostic FAIL is the current blocker; do not build the
planner until the action branch is healthy (RMS ~0.1+). Camera relocation / odometry conditioning is a
**fallback** if Run 002 still fails, no longer the primary plan.
