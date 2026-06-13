#!/usr/bin/env python3
"""Copy + downsize the figures the write-up uses into docs/assets/.

Manifest-driven and repeatable: re-run after regenerating any source figure.
PNGs wider than MAX_W are downscaled (LANCZOS); MP4s are copied as-is (all <1 MB).
Missing sources warn instead of failing so the site build never blocks.
"""
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path("/workspace/results")
OUT = REPO / "docs" / "assets"
MAX_W = 1600

# (source, dest name) — dest lands in docs/assets/
MANIFEST = [
    # §2 robot and data
    (REPO / "viz/world_trajectories.png", "world_trajectories.png"),
    (REPO / "viz/chunk_deltas.png", "chunk_deltas.png"),
    # §3 world model rollouts
    (RESULTS / "long_rollout_step8000_L12/long_0_cmp.mp4", "long_0_cmp.mp4"),
    # §5 signal debugging
    (REPO / "viz/signal-fsweep/f10/chunk_distributions.png", "fsweep_chunk_distributions.png"),
    (REPO / "viz/stationary-vs-translation/f10/latent_compare.png", "stationary_latent_compare.png"),
    # §6 run 002 diagnostics
    (RESULTS / "action_diag_run002_step4125/action_diagnostic.png", "action_diagnostic.png"),
    (RESULTS / "motion_rollout_run002_step4125/rotation_0_cmp.mp4", "rotation_0_cmp.mp4"),
    (RESULTS / "motion_rollout_run002_step4125/translation_0_cmp.mp4", "translation_0_cmp.mp4"),
    # §8 open-loop replay
    (REPO / "viz/lekiwi_6b1/synthetic_square.png", "replay_filmstrip.png"),
    # §9 camera ⊗ objective
    (RESULTS / "dist_sweep/curve.png", "dist_sweep_curve.png"),
    # §11 semantic pivot
    (RESULTS / "c1_smoke_strip.png", "c1_smoke_strip.png"),
    # §13 subgoal graph
    (RESULTS / "subgoal_graph/route_row0_nearchair.png", "route_montage.png"),
    (RESULTS / "subgoal_graph/goal_panel_preview.png", "goal_panel.png"),
    # §14 graph on robot
    (RESULTS / "subgoal_graph/route_strip_subgoals_preview.png", "route_strip_subgoals.png"),
]


def place(src: Path, dest: Path) -> str:
    if src.suffix == ".png":
        img = Image.open(src)
        if img.width > MAX_W:
            img = img.resize((MAX_W, round(img.height * MAX_W / img.width)), Image.LANCZOS)
            img.save(dest, optimize=True)
            return f"resized {img.width}x{img.height}"
        shutil.copyfile(src, dest)
        return "copied"
    shutil.copyfile(src, dest)
    return "copied"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = 0
    for src, name in MANIFEST:
        dest = OUT / name
        if not src.exists():
            print(f"  MISSING  {src}")
            missing += 1
            continue
        how = place(src, dest)
        print(f"  {how:>22}  {name}  ({dest.stat().st_size // 1024} KB)")
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"docs/assets: {total:.1f} MB total, {missing} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
