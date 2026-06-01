# Training Runs

Append-only telemetry log for **training runs**, maintained primarily by the pod-side agent (see
[[runpod-operator-guide]]). This is distinct from [[experiment-log]] (design-decision chronology):
this file is the operational record of *what was trained, on what, and how it went*.

Add a new entry at the top for each run, using the template below. Keep entries factual; link wandb.

---

## Entry template (copy for each run)

```
## Run <id> — <YYYY-MM-DD>

**Status:** running | completed | failed | aborted

### Setup
- NanoWM fork SHA: <git sha>
- NanoNAV SHA: <git sha>
- Dataset: kaushikpraka/wm-smallarea_nav30 (LeRobot v2.1, 30 Hz)  |  build SHA: <sha>
- Model: nanowm_b2 (B/2, SD-VAE, v-pred, additive)
- frame_interval: 5   action_aggregation: integrate_se2   action_dim: 2
- Effective batch: 64   batch_size: <n>   grad_accum: <n>   lr: 1e-4
- max_steps: <n>   pod: 1× H100 80 GB
- wandb: <url>

### Progress
- step <n>: loss <v>, <steps/sec>, action-embed RMS <v>
- checkpoints: <paths in $RESULTS_DIR>

### Anomalies / interventions
- <timestamp> <what happened> → <action taken>

### Table 5/6 diagnostic (gate)
- GT latent-L2:     <v>
- zero latent-L2:   <v>
- random latent-L2: <v>
- action-embed RMS: <v>
- verdict: PASS | FAIL
- notes: <e.g. GT clearly < zero/random; RMS ~0.1+>

### Outcome / next
- <result, decision, next action>
```

---

<!-- New run entries go below this line, newest first. -->

_No runs recorded yet. The first entry will be the initial NanoWM-B/2 checkpoint on
`wm-smallarea_nav30` (Stage 4 in [[roadmap]])._
