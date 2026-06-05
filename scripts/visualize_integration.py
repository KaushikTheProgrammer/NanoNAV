"""
Validate SE(2) pose integration on the real LeKiwi velocities.

Reads the dataset's tabular parquet (velocities only, no video), integrates with
scripts/nav_integration.py, and writes validation figures to viz/:

  1. world_trajectories.png  — per-episode world-frame paths (the integrated poses)
  2. units_check.png         — same episode integrated as deg/s vs rad/s (unit sanity)
  3. chunk_deltas.png        — per-chunk (Δx, Δθ) for f=5, and Δy magnitude (drop check)
  4. delta_distributions.png — dataset-wide (Δx, Δθ) distributions

Usage:
    .venv/bin/python scripts/visualize_integration.py \
        --parquet data/_cache/data.parquet --f 5 --out viz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nav_integration import base_velocities, integrate_trajectory, body_frame_chunk_deltas

DT = 1.0 / 30.0


def load_episodes(parquet: str):
    df = pd.read_parquet(parquet)
    A = np.stack(df["action"].to_numpy())  # [N, 9]
    eps = {}
    for ep in sorted(df["episode_index"].unique()):
        idx = (df["episode_index"] == ep).to_numpy()
        vx, omega = base_velocities(A[idx])
        eps[int(ep)] = (vx, omega)
    return eps


def fig_world_trajectories(eps, out: Path, n=12, n_arrows=8):
    from matplotlib.collections import LineCollection

    ids = list(eps)[:n]
    # Integrate once; gather a shared coordinate frame so every subplot is directly comparable.
    trajs, tmax = {}, 0.0
    gmin, gmax = np.inf, -np.inf
    for ep in ids:
        vx, omega = eps[ep]
        p = integrate_trajectory(vx, omega, DT, theta_in_degrees=True)
        trajs[ep] = p
        gmin = min(gmin, p[:, :2].min())
        gmax = max(gmax, p[:, :2].max())
        tmax = max(tmax, (len(vx)) * DT)
    pad = 0.05 * (gmax - gmin)
    lo, hi = gmin - pad, gmax + pad          # identical square limits for all subplots
    arrow_len = 0.03 * (hi - lo)             # heading-arrow length scaled to the shared frame

    cmap = plt.cm.viridis
    norm = plt.Normalize(0, tmax)
    cols, rows = 4, int(np.ceil(len(ids) / 4))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, ep in zip(axes, ids):
        p = trajs[ep]
        x, y, th = p[:, 0], p[:, 1], p[:, 2]
        t = np.arange(len(x)) * DT

        # path colored by time
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.0)
        lc.set_array(t[:-1])
        ax.add_collection(lc)

        # bold heading arrows (quiver), a handful evenly spaced along the path
        ks = np.linspace(0, len(x) - 1, n_arrows + 2).astype(int)[1:-1]
        ax.quiver(x[ks], y[ks], np.cos(th[ks]), np.sin(th[ks]),
                  color="crimson", angles="xy", scale_units="xy",
                  scale=1.0 / arrow_len, width=0.005, headwidth=4, headlength=5,
                  zorder=6, alpha=0.9)

        ax.scatter([0], [0], c="lime", s=60, ec="k", zorder=7, label="start")
        ax.scatter([x[-1]], [y[-1]], c="red", s=60, ec="k", marker="s", zorder=7, label="end")
        ax.set_title(f"ep {ep}  ({len(x)-1} steps, {t[-1]:.1f}s)", fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)     # standardized grid
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    for ax in axes[len(ids):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Integrated world-frame trajectories (heading = crimson arrows, color = time)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    # shared time colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label="time (s)")
    fig.savefig(out / "world_trajectories.png", dpi=110)
    plt.close(fig)


def fig_units_check(eps, out: Path, ep=None):
    ep = ep if ep is not None else list(eps)[0]
    vx, omega = eps[ep]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, deg, title in [
        (axes[0], True, "theta.vel = DEGREES/s (expected)"),
        (axes[1], False, "theta.vel = RADIANS/s (wrong → spins)"),
    ]:
        p = integrate_trajectory(vx, omega, DT, theta_in_degrees=deg)
        ax.plot(p[:, 0], p[:, 1], lw=1.0)
        ax.scatter([0], [0], c="g", s=30); ax.scatter([p[-1, 0]], [p[-1, 1]], c="r", s=30)
        total_turn = np.rad2deg(abs(p[-1, 2]))
        ax.set_title(f"{title}\ntotal |heading change| = {total_turn:.0f}°")
        ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)
    fig.suptitle(f"Unit sanity check — episode {ep}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "units_check.png", dpi=110)
    plt.close(fig)


def fig_chunk_deltas(eps, out: Path, f, ep=None):
    ep = ep if ep is not None else list(eps)[0]
    vx, omega = eps[ep]
    d = body_frame_chunk_deltas(vx, omega, DT, f, theta_in_degrees=True)
    dx, dy, dth = d[:, 0], d[:, 1], d[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(dx * 1000, label="Δx (mm)"); axes[0].plot(np.rad2deg(dth), label="Δθ (deg)")
    axes[0].set_xlabel(f"chunk (f={f})"); axes[0].set_title(f"ep {ep}: per-chunk action"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(np.abs(dy) * 1000, c="purple")
    axes[1].set_xlabel("chunk"); axes[1].set_ylabel("|Δy| (mm)")
    axes[1].set_title(f"Dropped Δy magnitude (max {np.abs(dy).max()*1000:.2f} mm)"); axes[1].grid(alpha=0.3)
    axes[2].scatter(dx * 100, np.rad2deg(dth), s=10, alpha=0.5)
    axes[2].set_xlabel("Δx (cm)"); axes[2].set_ylabel("Δθ (deg)")
    axes[2].set_title("action scatter"); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "chunk_deltas.png", dpi=110); plt.close(fig)


def fig_delta_distributions(eps, out: Path, f):
    alld = np.concatenate([body_frame_chunk_deltas(vx, om, DT, f, True) for vx, om in eps.values()])
    dx, dy, dth = alld[:, 0], alld[:, 1], alld[:, 2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(dx * 100, bins=60); axes[0].set_xlabel("Δx (cm)"); axes[0].set_title(f"Δx  (mean {dx.mean()*100:.2f} cm)")
    axes[1].hist(np.rad2deg(dth), bins=60); axes[1].set_xlabel("Δθ (deg)"); axes[1].set_title(f"Δθ  (std {np.rad2deg(dth).std():.1f}°)")
    axes[2].hist(np.abs(dy) * 1000, bins=60); axes[2].set_xlabel("|Δy| (mm)"); axes[2].set_title(f"|Δy|  (99pct {np.percentile(np.abs(dy)*1000,99):.2f} mm)")
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle(f"Dataset-wide per-chunk deltas (f={f}, {len(alld)} chunks)", fontsize=14)
    fig.tight_layout(); fig.savefig(out / "delta_distributions.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/_cache/data.parquet")
    ap.add_argument("--f", type=int, default=5)
    ap.add_argument("--out", default="viz")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    eps = load_episodes(args.parquet)
    print(f"loaded {len(eps)} episodes")
    fig_world_trajectories(eps, out)
    fig_units_check(eps, out)
    fig_chunk_deltas(eps, out, args.f)
    fig_delta_distributions(eps, out, args.f)

    # console summary
    alld = np.concatenate([body_frame_chunk_deltas(vx, om, DT, args.f, True) for vx, om in eps.values()])
    print(f"chunks (f={args.f}): {len(alld)}")
    print(f"  Δx : mean {alld[:,0].mean()*100:6.3f} cm   max {alld[:,0].max()*100:6.3f} cm")
    print(f"  Δθ : std  {np.rad2deg(alld[:,2]).std():6.2f} deg  range [{np.rad2deg(alld[:,2]).min():.1f}, {np.rad2deg(alld[:,2]).max():.1f}]")
    print(f"  |Δy|: max  {np.abs(alld[:,1]).max()*1000:6.3f} mm  99pct {np.percentile(np.abs(alld[:,1])*1000,99):.3f} mm")
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
