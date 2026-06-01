"""
SE(2) pose integration for LeKiwi base velocities — the single source of truth.

Both the offline dataset builder (`build_lekiwi_nav_dataset.py`) and the NanoWM
`integrate_se2` dataloader patch must mirror THIS math, so the (Δx, Δθ) the model
trains on is exactly what the visualizer validates.

LeKiwi base action layout (9-D): [6 arm joints, x.vel, y.vel, theta.vel].
  - x.vel:     forward linear velocity, metres/second
  - y.vel:     lateral (strafe) — 0 by construction (no strafe binding)
  - theta.vel: yaw rate. LeKiwi's lerobot teleop emits this in DEGREES/second,
               so it must be converted to rad/s before integration. Pass
               theta_in_degrees=True (default).

Unicycle kinematics, integrated step-by-step (matches context/action-representation.md):
    x += v*dt*cos(theta); y += v*dt*sin(theta); theta += omega*dt
"""

import numpy as np

# Base-velocity indices within the 9-D LeKiwi action vector.
IDX_VX = 6
IDX_VY = 7
IDX_VTHETA = 8


def base_velocities(action_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice (v_x [m/s], omega_raw) out of the full 9-D action array [..., N, 9]."""
    a = np.asarray(action_9d, dtype=np.float64)
    return a[..., IDX_VX], a[..., IDX_VTHETA]


def integrate_trajectory(
    vx: np.ndarray,
    omega: np.ndarray,
    dt: float = 1.0 / 30.0,
    theta_in_degrees: bool = True,
) -> np.ndarray:
    """Continuous WORLD-frame pose integration over a whole sequence.

    Returns poses of shape [N+1, 3] = (x, y, theta_rad), starting at the origin
    (0, 0, 0). This is the robot's path — used for visualization/validation.
    """
    vx = np.asarray(vx, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if theta_in_degrees:
        omega = np.deg2rad(omega)

    n = len(vx)
    poses = np.zeros((n + 1, 3), dtype=np.float64)
    x = y = th = 0.0
    for i in range(n):
        x += vx[i] * dt * np.cos(th)
        y += vx[i] * dt * np.sin(th)
        th += omega[i] * dt
        poses[i + 1] = (x, y, th)
    return poses


def body_frame_chunk_deltas(
    vx: np.ndarray,
    omega: np.ndarray,
    dt: float = 1.0 / 30.0,
    f: int = 5,
    theta_in_degrees: bool = True,
) -> np.ndarray:
    """Per-chunk BODY-frame displacement — the model action.

    Splits the sequence into consecutive windows of `f` steps. Each window plants
    a fresh local frame (origin, heading 0) and integrates the unicycle model over
    its `f` velocities. Returns [M, 3] = (Δx, Δy, Δθ_rad), M = floor(N / f).

    The model uses (Δx, Δθ); Δy is returned only so the visualizer can confirm it
    is negligible (the "drop Δy" assumption in context/action-representation.md).
    """
    vx = np.asarray(vx, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if theta_in_degrees:
        omega = np.deg2rad(omega)

    n = len(vx)
    m = n // f
    out = np.zeros((m, 3), dtype=np.float64)
    for c in range(m):
        x = y = th = 0.0
        for i in range(c * f, c * f + f):
            x += vx[i] * dt * np.cos(th)
            y += vx[i] * dt * np.sin(th)
            th += omega[i] * dt
        out[c] = (x, y, th)
    return out
