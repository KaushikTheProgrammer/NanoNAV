#!/usr/bin/env bash
# Launch a closed-loop MPC run with baked-in presets.
#
#   scripts/run_mpc.sh <graph|nograph|sdvae> <goal> [extra lekiwi_mpc.py args...]
#
#   graph    semantic WM (DINO tokens) + subgoal-graph navigation
#   nograph  semantic WM, direct goal pursuit (no graph)
#   sdvae    original SD-VAE world model (run 002, step-8000)
#
# <goal> is a name under goals/ (e.g. nearhamper1) or a path to a goal .png.
# Output .rrd/.log auto-increment under /workspace/results, never overwriting.
# Extra args pass through (e.g. --reach-thresh 0.15 --max-steps 200).
# DRY=1 prints the command instead of launching.
set -euo pipefail

MODE="${1:?usage: run_mpc.sh <graph|nograph|sdvae> <goal> [extra args...]}"
GOAL_ARG="${2:?usage: run_mpc.sh <graph|nograph|sdvae> <goal> [extra args...]}"
shift 2

REPO=/workspace/NanoNAV
RESULTS=/workspace/results
PY=/workspace/nanowm-venv/bin/python
SEMANTIC_CKPT=$RESULTS/20260610_112629-C0ext-dinoB1-x0-adalnfuse-F4S10-lekiwi/checkpoints/across_timesteps/epoch=19-step=12000.ckpt
SDVAE_CKPT=$RESULTS/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt

# resolve goal: name under goals/ (dir with goal.png, or flat .png) or explicit path
if [[ -f "$GOAL_ARG" ]]; then GOAL="$GOAL_ARG"; GOAL_NAME=$(basename "${GOAL_ARG%.*}")
elif [[ -f "$REPO/goals/$GOAL_ARG/goal.png" ]]; then GOAL="goals/$GOAL_ARG/goal.png"; GOAL_NAME="$GOAL_ARG"
elif [[ -f "$REPO/goals/$GOAL_ARG.png" ]]; then GOAL="goals/$GOAL_ARG.png"; GOAL_NAME="$GOAL_ARG"
else
    echo "goal '$GOAL_ARG' not found (looked for the path itself, goals/$GOAL_ARG/goal.png, goals/$GOAL_ARG.png)" >&2
    echo "available goals:" >&2; ls "$REPO/goals" >&2; exit 1
fi

ENV_PREFIX=()
case "$MODE" in
    graph)
        ARGS=(--ckpt "$SEMANTIC_CKPT" --graph "$RESULTS/subgoal_graph"
              --token-decoder "$RESULTS/token_decoder/decoder.pt" --reach-thresh 0.08)
        ENV_PREFIX=(WEBDINO_MODEL_PATH=facebook/dinov2-small)
        TAG="semantic_graph" ;;
    nograph)
        ARGS=(--ckpt "$SEMANTIC_CKPT"
              --token-decoder "$RESULTS/token_decoder/decoder.pt" --reach-thresh 0.08)
        ENV_PREFIX=(WEBDINO_MODEL_PATH=facebook/dinov2-small)
        TAG="semantic_nograph" ;;
    sdvae)
        ARGS=(--ckpt "$SDVAE_CKPT" --reach-thresh 25)
        TAG="sdvae" ;;
    *) echo "unknown mode '$MODE' (graph|nograph|sdvae)" >&2; exit 1 ;;
esac

# auto-increment run name: mpc_<tag>_<goal>, _<goal>2, _<goal>3, ...
BASE="mpc_${TAG}_${GOAL_NAME}"
N=""
while [[ -e "$RESULTS/${BASE}${N}.rrd" || -e "$RESULTS/${BASE}${N}.log" ]]; do N=$(( ${N:-1} + 1 )); done
RRD="$RESULTS/${BASE}${N}.rrd"; LOG="$RESULTS/${BASE}${N}.log"

CMD=(env "${ENV_PREFIX[@]}" "$PY" scripts/lekiwi_mpc.py
     --planner wm --ip 127.0.0.1 --nanowm-src external/nanowm/src
     --goal "$GOAL" --max-steps 150 --speed-scale 1.0
     --rerun-web --rerun-save "$RRD"
     "${ARGS[@]}" "$@")

if [[ "${DRY:-0}" == "1" ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

cd "$REPO"
echo go | nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "launched $MODE run -> $GOAL_NAME (pid $PID)"
echo "  log:  $LOG"
echo "  rrd:  $RRD"
echo "  tail: tail -f $LOG"
