# Run 002 — cross-checkpoint rollout evaluation

Seeded (seed 42) rollout eval of NanoWM-B/2 Run 002 (f=10) at training steps **4125 (val-best) / 6K /
8K / 10K / 12K**, to answer *does more training improve rollout quality* without trusting the (weak,
for diffusion-forcing) val_loss. Per checkpoint: the action-conditioning gate
(`action_diagnostic.py`, GT/zero/random, 16 batches) + motion-selected GT-vs-pred rollouts
(`motion_rollout_viz.py`, same deterministic translation/rotation/arc chunks, grids + `_cmp.mp4`
videos). Full outputs (grids + videos + JSON) live at `/workspace/results/eval_run002/`.

## Result — rollout quality is U-shaped in training step

See `summary_plot.png` / `summary_table.md`.

- **Prediction accuracy (GT latent-L2) and motion-tracking improve *past* the val-best (4125), peak at
  ~step 6K–8K, then overfitting degrades them through 12K.** So the val-loss optimum is NOT the best
  rollout model, and training to 12K overshoots.
- Best GT accuracy + translation + arc at **step 8000**; best rotation + action-separation at **step 6000**.
- Action separation (random−GT) holds ~10 throughout; action-embed RMS only creeps 0.0089→0.0102
  (still ≪ the 0.05 gate) — the action branch is robust; the **RMS gate is mis-calibrated** for the 2-D
  additive embedder (see `context/training.md`).

**⇒ carry step-8000 into the CEM/MPC planner** (Stage 6) — not the val-best (4125) nor the final (12000).
**Key methodological point: val_loss mis-ranked the checkpoints** (called 4125 optimal; rollouts say
~8K) — judging by rollouts was decisive. See `context/training-runs.md` (Run 002) + `context/experiment-log.md`.
