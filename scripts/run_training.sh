#!/usr/bin/env bash
# Launch NanoWM-B/2 training on the LeKiwi nav dataset (RunPod H100).
# Env is the uv venv at /workspace/nanowm-venv (not conda — see context/runpod-setup.md notes).
# Usage: tmux new-session -d -s train 'bash /workspace/NanoNAV/scripts/run_training.sh'
# NOTE: no `set -u` — the venv activate script references unbound vars (PS1, etc.).
set -eo pipefail

export WORKDIR=/workspace
export REPO_DIR=/workspace/NanoNAV
export RESULTS_DIR=/workspace/results
export LEKIWI_DATA_ROOT=/workspace/data/lekiwi
export WANDB_PROJECT=nanonav
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RESULTS_DIR"

# Persistent secrets (WANDB_API_KEY, etc.) — lives on /workspace so it survives pod restarts.
# Only /workspace persists; the root FS ~/.netrc wandb login does NOT (caused Run 002's first crash).
if [ -f /workspace/secrets/env.sh ]; then
    source /workspace/secrets/env.sh
else
    echo "[warn] /workspace/secrets/env.sh not found — wandb may fail to authenticate" >&2
fi

source /workspace/nanowm-venv/bin/activate
cd "$REPO_DIR/external/nanowm"

python -c "import pytorch_lightning as pl, torch; print('[env] PL', pl.__version__, '| torch', torch.__version__)"

# Optional Lightning native resume: set RESUME_CKPT=/path/to/x.ckpt to continue a run with full
# optimizer/LR/step state (distinct from warm-start). Logs to train_resume.log so the original
# train.log is preserved.
RESUME_ARG=()
LOG="$RESULTS_DIR/train.log"
if [ -n "${RESUME_CKPT:-}" ]; then
    RESUME_ARG=("experiment.ckpt_path=$RESUME_CKPT")
    LOG="$RESULTS_DIR/train_resume.log"
    echo "[resume] continuing from $RESUME_CKPT"
fi

# python -u so the log streams live (tail -f it to monitor). tee (not -a) = fresh log each launch.
exec python -u src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2 \
    "${RESUME_ARG[@]}" \
    2>&1 | tee "$LOG"
