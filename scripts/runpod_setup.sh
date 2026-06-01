#!/usr/bin/env bash
# RunPod H100 setup + launch for NanoNAV (NanoWM-B/2 on LeKiwi nav data).
#
# Run this ON THE POD after provisioning (UI):
#   * 1x H100 80GB, PyTorch/CUDA template
#   * a persistent / network volume mounted at /workspace  (so dataset + checkpoints survive restarts)
#
# It clones NanoNAV WITH its submodule (external/nanowm = the patched NanoWM fork), builds the
# derived v2.1 dataset onto the volume, and launches training under tmux + wandb. Idempotent-ish:
# re-running skips the dataset build if it already exists.
#
# Usage:
#   bash runpod_setup.sh
# Override any of these first if needed:
#   NANONAV_URL=...  RESULTS_DIR=...  LEKIWI_DATA_ROOT=...  WANDB_PROJECT=...
set -euo pipefail

# ---- config -----------------------------------------------------------------
NANONAV_URL="${NANONAV_URL:-https://github.com/KaushikTheProgrammer/NanoNAV.git}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKDIR/NanoNAV}"
export RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results}"
export LEKIWI_DATA_ROOT="${LEKIWI_DATA_ROOT:-$WORKDIR/data/lekiwi}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanonav}"
ENV_NAME="${ENV_NAME:-nanowm}"

mkdir -p "$RESULTS_DIR" "$(dirname "$LEKIWI_DATA_ROOT")"

# ---- 1. clone repo + submodule (the NanoWM fork) ----------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --recurse-submodules "$NANONAV_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" pull --recurse-submodules
  git -C "$REPO_DIR" submodule update --init --recursive
fi
NANOWM_DIR="$REPO_DIR/external/nanowm"
[ -f "$NANOWM_DIR/environment.yml" ] || { echo "ERROR: submodule external/nanowm not populated"; exit 1; }

# ---- 2. conda env (NanoWM pins lerobot-datasets==2.1.0) ----------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda env create -f "$NANOWM_DIR/environment.yml" -n "$ENV_NAME"
fi
conda activate "$ENV_NAME"
# extra deps for the preprocessing script (raw v3.0 read): PyAV for av1 decode + hf hub
pip install --quiet av huggingface_hub

# ---- 3. auth (interactive: paste tokens once) -------------------------------
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login
wandb status >/dev/null 2>&1 || wandb login || true   # wandb optional; training still runs

# ---- 4. build derived dataset onto the volume (skip if present) -------------
# CPU-bound av1 decode, ~10-20 min. Writes a LeRobot v2.1 dataset NanoWM reads natively.
if [ ! -d "$LEKIWI_DATA_ROOT/meta" ]; then
  echo "Building derived dataset -> $LEKIWI_DATA_ROOT"
  ( cd "$REPO_DIR" && python scripts/build_lekiwi_nav_dataset.py --limit 2 --dry-run )   # smoke test
  ( cd "$REPO_DIR" && python scripts/build_lekiwi_nav_dataset.py --out-root "$LEKIWI_DATA_ROOT" )
else
  echo "Dataset already present at $LEKIWI_DATA_ROOT, skipping build."
fi

# ---- 5. launch training under tmux ------------------------------------------
# eff-bs 64 on one H100; adjust batch_size/grad-accum in configs/experiment/lekiwi_nav.yaml if OOM.
LAUNCH="cd '$NANOWM_DIR' && conda activate $ENV_NAME && \
RESULTS_DIR='$RESULTS_DIR' LEKIWI_DATA_ROOT='$LEKIWI_DATA_ROOT' WANDB_PROJECT='$WANDB_PROJECT' \
python src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2"

if tmux has-session -t train 2>/dev/null; then
  echo "tmux session 'train' already running. Attach with: tmux attach -t train"
else
  tmux new-session -d -s train "$LAUNCH 2>&1 | tee -a '$RESULTS_DIR/train.log'"
  echo "Training launched in tmux session 'train'."
fi

cat <<EOF

----------------------------------------------------------------
Setup complete.
  Repo:     $REPO_DIR  (NanoWM fork at external/nanowm)
  Dataset:  $LEKIWI_DATA_ROOT
  Results:  $RESULTS_DIR   (log: $RESULTS_DIR/train.log)
  wandb:    project '$WANDB_PROJECT'

  Watch:    tmux attach -t train       (detach: Ctrl-b d)
  Tail log: tail -f $RESULTS_DIR/train.log
  GPU:      watch -n2 nvidia-smi

A pod-side Claude session should follow context/runpod-operator-guide.md and log the run in
context/training-runs.md.
----------------------------------------------------------------
EOF
