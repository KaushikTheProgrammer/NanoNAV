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
- **3b — Validation:** load + decode + action-range sanity ✅ (vx∈[0,0.1] m/s, ω∈[−0.524,0.524] rad/s
  = ±30°/s = ±π/6, the true range measured across all 50 episodes on 2026-06-04; an earlier
  "[−0.32,0.34]" figure undercounted the max).
  SD-VAE `compare` of frame *k* vs *k+f* (visual-flow vs `(Δx, Δθ)`) still ⬜ — no independent
  odometry exists (state is velocity, not pose).

## ✅ Stage 4 — First Checkpoint (trained; overfit early — see Stage 5)

NanoWM-B/2, v-prediction, additive injection, SD-VAE. Integrated `(Δx, Δθ)` action via the
`integrate_se2` dataloader patch; `frame_interval=5` (the tunable reach knob). Trained on a single
**RunPod H100** (eff-bs 64, f=5). **Run 001** (wandb `x3ub`) trained on the uv-venv stack, but
**overfit by epoch ~3** (50 episodes is tiny for B/2; 50K steps = ~81 epochs, and the config saved no
best-val checkpoint) and was stopped at ~23K steps. See [[runpod-setup]], [[training-runs]].

## ▶️ Stage 5 — Action-Conditioning Diagnostic (Table 5/6) — **Run 002 trained; action branch alive, re-gating via rollouts**

`action_diagnostic.py` (GT / zero / random rollouts): GT latent-L2 must clearly beat zero/random and
action-embedding RMS must be ~0.1+. **Run 001 (overfit f=5 step-10K): FAIL** — RMS **0.0088**, GT 37.8
vs zero 42.0 / random 42.4 (zero≈random ⇒ action ignored).

**Run 002 (f=10, trained to 12K, gate on the best-val step-4125 ckpt):** GT **36.1**, zero **40.7**,
random **45.2**, RMS **0.0089**. The legacy RMS gate still reads **FAIL**, but the rollout signal is
**materially healthier**: a clean, widening **gt < zero < random** separation (random now distinctly
worse than zero — the model uses action *content*), and motion rollouts visibly track real
translation/rotation/arc. The RMS being ~identical to Run 001 across two very different checkpoints ⇒
**RMS is mis-calibrated / architecturally pinned** for the 2-D additive embedder, not a live signal —
the separation + motion-tracking are the meaningful gate. (Earlier "translation unobservable" claim was
refuted: translation IS observable, `viz/stationary-vs-translation/`; the camera was never the problem.)

**Cross-checkpoint rollout eval — DONE** (seeded, 4125/6K/8K/10K/12K): rollout quality is **U-shaped in
step — peaks at ~6K–8K then overfits** (the val-best 4125 is *not* the best rollout model; 12K
overshoots). Action separation stays ~10 throughout; RMS ~0.009–0.010 (gate mis-calibrated). **⇒
step-8000 is the checkpoint to carry into Stage 6** (best GT accuracy + translation + arc). val_loss
mis-ranked the checkpoints, so judging by rollouts was decisive. See [[training-runs]] (Run 002),
[[open-questions]], [[experiment-log]].

## ▶️ Stage 6 — Short-Range Planner (CEM/MPC) — **6a DONE (PASS); 6b in progress (first closed-loop run done — planning works, convergence open)**

The CEM/MPC core already exists (`cem_planner.py` `CEMPlanner`, `diffusion_world_model.rollout`,
`objective.py`, `preprocessor.py`, `planning_experiment.py` + `_sample_dset_goals`). Stage 6 is **wiring
it for LeKiwi**: the `envs/` dir has no LeKiwi/dataset env. Plan (eval-grounded, see [[planning]] "Stage
6 — Implementation Plan"):
- **6a — offline CEM eval — ✅ DONE (2026-06-04, PASS).** Standalone `src/sample/offline_planning_eval.py`
  (NOT a registry env — LeKiwi has no simulator/`states.pth` layout, so the sim-coupled
  `PlanningExperiment._run_mpc` doesn't fit) + `configs/planning/lekiwi.yaml` (6b scaffold). CEM recovers a
  goal-reaching action sequence to a val frame `goal_H=3` chunks ahead, graded against the dataset answer
  key. **Result on step-8000, 35 stratified val scenes × DDIM {20,5,3}: all four gates pass** — CEM beats
  `do_nothing` 100%, `reached_ratio` ~1.0–1.1 (WM-optimal) in every motion bucket, action sign 100% / dxErr
  ~1 cm / dθErr ~2.5°, decoded montages land on the goal, and **DDIM=3 holds with no pivot collapse** (the
  cheap-sampler concern did not materialize — `cem_reached` even dropped slightly). The residual goal gap is
  WM prediction error, not planner failure. ⇒ the ~7 s/replan DDIM=3 / 32×3 regime is confirmed; the engine
  is validated. Open-loop accuracy only — closed-loop is 6b. See [[planning]] "6a — RESULTS",
  `results/offline_planning_step8000/`.
- **6b — closed-loop on LeKiwi — SPEC'D (next), ready to implement; needs the robot.** RunPod runs the
  lerobot `LeKiwiClient` (the Pi keeps the already-working host) over **Tailscale** — so lerobot's ZMQ
  transport IS the obs/command channel and CEM inference is a local call in the same process (no bespoke
  inference API). Stop-and-plan MPC wraps the 6a-validated engine (step-8000, DDIM=3, 32×3, H=3,
  replan-every-chunk); goals are real `top` frames (drive-and-snapshot / pre-staged); live telemetry via
  **rerun-over-Tailscale** to the Mac viewer, rendering the winning **and the selected top-K elite** WM
  rollouts (needs a small backward-compat `CEMPlanner` patch to surface elites). Sub-steps **6b.0
  transport+units ✅ DONE** (Pi 10.0.0.125: RTT ~15 ms; `x.vel` m/s +fwd; `theta.vel` deg/s +CCW = no sign
  flip, convert ω×57.3; low-speed turn deadband noted) → **6b.1 open-loop replay ✅ DONE** (trajectories match
  on hardware; chunk-collapse ~0 cm even on a 117° arc; per-chunk timing pinned to 333 ms; action range
  corrected to ±30°/s; dataset read direct from parquet+mp4 since recent lerobot can't load v2.1) →
  **6b.3 controller harness ✅ validated on hardware with a stub planner** (`scripts/lekiwi_mpc.py`:
  stop-and-plan loop + precise timing + clamp + termination + rerun 0.26 telemetry, planner injected) →
  **6b.2 engine ✅ validated on the pod** (fork `4720053` `src/sample/lekiwi_engine.LekiwiPlanner` wraps the 6a
  path + live-frame letterbox preprocess + `CEMPlanner` `return_elites` patch; **smoke-test PASS on H100 +
  step-8000**, 2026-06-05 — raw 480×640 frames through the full preprocess→encode→CEM→rollout→decode path;
  action stats match, do_nothing dist≈0, and on a moving pair (ep11 504→534, GT fwd+right-turn) CEM recovered
  the **correct signs** `vx=+0.067, θ=−15.6°/s`; decoded `imagined` is a coherent top-view. **Hard
  precondition surfaced:** the live controller MUST pass `action_mean/std` explicitly — the engine's
  dataset-reconstruction fallback is dead on the pod (private-repo 401 + lerobot-v3-can't-read-v2.1) and the
  stats aren't in the ckpt; see [[experiment-log]] 2026-06-05) →
  **6b.4 goal capture ✅ tool built** (`scripts/capture_goal.py`: snapshot the `top` frame → goal.png +
  256² letterbox preview matching the planner; runs on the Mac, no GPU) →
  **6b.3 FIRST live WM closed-loop run ▶️ PARTIAL (2026-06-05):** planning worked on the real robot (correct
  stats, sane CEM commands, ~7.5 s/plan @ DDIM=3, stop-and-plan loop executed) but **did NOT converge**
  (`dist_to_goal` ~44–46 over 22 steps, reach-thresh 35) and the **Pi host + SSH tunnel dropped mid-run**;
  rerun live-viz fixed via **`--rerun-web`** (pod-hosted web viewer, 9090/9877) after the `-R 9876` path
  collided with VS Code's Mac port — `.rrd` captured. Next: redo with live viz + diagnose the `dist` plateau
  (WM under-drive vs goal beyond horizon-3 reach vs the truncated run). See [[experiment-log]] 2026-06-05,
  [[tailscale-setup]] "Live rerun telemetry". → 6b.5 telemetry. **Closed-loop needs a
  pod↔robot bridge** ([[tailscale-setup]]): **recommended = SSH reverse tunnel over RunPod's exposed TCP/SSH
  port** (`ssh -N -R 5555/-R 5556` from the Mac → pod runs `lekiwi_mpc.py --planner wm --ip 127.0.0.1`) — no
  TUN, no new code, reuses the validated pod-as-client path. Tailscale kernel mode is **blocked** (pod has no
  `/dev/net/tun`, not privileged to create it); userspace Tailscale is fragile for ZMQ. Top trap: `theta.vel`
  deg/s↔rad/s (57× scale). **Develop locally for free** (all authoring + the no-model robot checks 6b.0/6b.1
  with the Mac as lerobot client on the LAN, stub-planner end-to-end test); **resume on the pod only for live
  CEM inference** (swap stub→real WM, Mac/LAN→RunPod/Tailscale — a config swap, not a rewrite). Full spec in
  [[planning]] "6b — Closed-Loop MPC on LeKiwi".
- **6c — long-range:** topological waypoint graph.

Params from the evals: **step-8000**, **H = 3–5 chunks** (reliable rollout window; at f=10 → ~10–17 cm
reach), latent-L2 scoring valid **<~30 cm**, CEM ~64×5×top-10, DDIM 20. Develop the code on a cheap box
(repo only); run on GPU here on demand (ckpt + dataset stay on `/workspace`). See [[planning]].

## ⬜ Stage 7 — Long-Range Navigation

Topological waypoint graph (+ DepthAnything3 metric edges) is the recommended start; HWM / learned
distance as alternatives. Where most [[open-questions]] cluster. See [[planning]].

## ⬜ Stage 8 — Extensions (future)

Pattern B comparison, real-time planning, mobile manipulation (arm), multi-room transfer, latent
actions. See [[open-questions]].

---

## Current critical path

✅ 3a (built) → ✅ 4 (Run 001 trained, overfit f=5) → ✅ **Run 002 trained to 12K at f=10**
(best-val checkpointing; 3 crashes fixed + pushed) → **▶️ 5: re-gating via rollouts** — the action
branch is now alive/action-sensitive (clean gt<zero<random + visible motion tracking), the legacy RMS
gate reads FAIL but is judged mis-calibrated; the **cross-checkpoint rollout eval** found rollout
quality peaks at **~6K–8K** then overfits ⇒ **step-8000 is the chosen planner checkpoint** → ✅ **6a
(offline planner eval) PASSED** (35 stratified val scenes × DDIM {20,5,3}: CEM WM-optimal in every motion
bucket, DDIM=3 holds, engine validated) → ✅ **6b.0/6b.1 hardware bring-up + 6b.3 harness + 6b.2 engine
smoke-test PASS** (LekiwiPlanner runs end-to-end on H100/step-8000; correct sign recovery on a moving pair;
explicit `action_mean/std` is now a hard launch precondition) → **next: 6b.3 closed-loop on LeKiwi** (swap
stub→`--planner wm`, needs the robot). Decision gate for the
planner is now **rollout health** (action separation + motion-tracking fidelity), not the RMS number. Camera
relocation / odometry conditioning remains a **fallback** only if rollouts prove inadequate.
