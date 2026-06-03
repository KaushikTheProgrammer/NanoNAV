# NanoWM × LeKiwi Navigation — Overview

## Project Goal

Use NanoWM (a diffusion-forcing world model) for goal-conditioned navigation on a LeKiwi mobile manipulator. The world model predicts future camera frames given current observations and actions. At inference, CEM planning searches for actions whose predicted futures match a goal image. The model learns general environment dynamics; tasks are specified at inference via goal images.

## Current Phase

**Phase 4→5: First checkpoint trained; Table 5/6 diagnostic FAILED — but the cause is a TRAINING
problem (overfit + low f), not a camera/observability one.** Dataset built (50 eps / 44,926 frames →
`/workspace/data/lekiwi`) and NanoWM-B/2 **Run 001** trained on a RunPod H100 (uv venv, `integrate_se2`,
f=5, eff-bs 64). It **overfit by epoch ~3** and the step-10K (epoch-16, overfit) checkpoint **failed
the gate** (action-embed RMS 0.0088 ≪ 0.1).

A **controlled stationary-vs-translation latent contrast** (2026-06-03; `stationary_vs_translation.py`,
`viz/stationary-vs-translation/`) **overturned the earlier "translation is geometrically unobservable"
conclusion**: holding rotation near zero, pure-translation chunks change the SD-VAE latent ~2× more than
stationary ones (AUC 0.94 @ f=5 → 0.98 @ f=10/20), with a clean dose-response and a near-field-floor
parallax footprint. The old `corr(|Δx|, latentL2)≈0` was an artifact of bang-bang Δx + pooled rotation
chunks; **raising f from 5→10 lifts translation's SNR over the noise floor from ~1:1 to ~1.6:1.**
**Next: Run 002 — retrain at f=10 with best-val checkpointing + low max_steps, then re-run the gate on
the best-val model.** Camera relocation / odometry conditioning is demoted to a fallback, not a
prerequisite. See [[training-runs]] (Run 001 + Run 002 plan), [[experiment-log]], [[open-questions]].

## Project Tracking

Progress, experiments, and decisions are tracked in **git** (this repo), not Obsidian. The `context/` notes are the living design record; the chronological record lives in [[experiment-log]].

## Key Decisions (Settled)

- **Architecture:** Pattern A — action-conditioned forward model + CEM/MPC planning (not Pattern B video-as-plan + IDM)
- **Backbone:** NanoWM-B/2 (160M params)
- **Latent space:** SD-VAE [4, 32, 32] — action-sensitive, decodable for visualization. DINO/V-JEPA fail at action conditioning (Finding #4)
- **Action representation:** Body-frame pose delta (Δx, Δθ), 2D. Integrated from f=5 velocity commands per chunk. Δy dropped (negligible at chunk timescales)
- **Camera:** Elevated third-person (~55° tilt) mounted on robot arm structure. Captures robot body (fixed reference), near floor, mid objects, far walls
- **Control:** Unicycle model, 2-DOF (v_x, ω). No strafe. Exact kinematics for pose tracking
- **Data paradigm:** General environment exploration, not task demonstrations. Random/diverse driving. Task enters only at inference via goal image
- **Objective:** v-prediction, cosine + ZTSNR schedule
- **Planning:** Stop-and-plan MPC. ~1-2s per replan with full 20 DDIM steps. Not real-time, acceptable for prototype

## File Index

- [[action-representation]] — Body-frame delta construction, normalization, camera geometry
- [[data-collection]] — Episode structure, controller setup, logging, dataset sizing
- [[training]] — Model config, hyperparameters, diagnostic protocol
- [[planning]] — CEM/MPC pipeline, scoring, waypoint scaffold, long-range solutions
- [[experiment-log]] — Chronological record of design decisions
- [[open-questions]] — Unresolved items and future directions
- [[roadmap]] — Staged execution plan with current status
- [[nanowm-integration]] — How the dataset plugs into NanoWM (concat vs integrate, v3.0→v2.1, the patch)
- [[runpod-setup]] — Bring-up runbook for the pod-side agent (install, clone, build, launch)
- [[runpod-operator-guide]] — Runbook for the pod-side agent that babysits training
- [[training-runs]] — Per-run training telemetry log
