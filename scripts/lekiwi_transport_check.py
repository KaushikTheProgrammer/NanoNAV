#!/usr/bin/env python3
"""
6b.0 — LeKiwi transport + units smoke test (NanoNAV Stage 6b).

Runs with your Mac as the lerobot CLIENT on the local LAN (your working teleop path):
NO Tailscale, NO GPU, NO world model. It confirms the robot-facing contract the
closed-loop MPC will depend on:

  1. LeKiwiClient connects to the Pi host and get_observation() returns a decodable
     `top` frame (the camera the world model was trained on).
  2. send_action() actually drives the base, and we learn the SIGN + UNITS of
     `x.vel` and `theta.vel` empirically.
  3. round-trip latency is sane.

────────────────────────────────────────────────────────────────────────────────────
THE UNITS QUESTION THIS ANSWERS  (the #1 silent bug in 6b)
  build_lekiwi_nav_dataset.py converted raw LeKiwi `theta.vel` from deg/s → rad/s for
  training, so the WORLD MODEL's ω is in rad/s. lerobot's send_action almost certainly
  expects the robot's NATIVE unit. If that unit is deg/s, the live controller MUST
  convert ω(rad/s) → deg/s (×180/π ≈ ×57.3) before send_action — otherwise every turn
  is off by ~57×. This script determines the native unit by a SAFE escalation:
    • command theta.vel = 0.3  (safe in BOTH hypotheses: 0.3 rad/s ≈ 17°/s, or 0.3°/s ≈ nothing)
        – visible rotation  → native unit is RAD/S  (no conversion needed)
        – no rotation       → native unit is DEG/S  → then a 12°/s test confirms + gives sign
  Forward `x.vel` is assumed m/s (lerobot native); we confirm magnitude + sign too.
────────────────────────────────────────────────────────────────────────────────────
SAFETY — READ BEFORE RUNNING
  • Motion is OFF by default. Pass --enable-motion to allow it, and the script still
    makes you confirm interactively that the WHEELS ARE OFF THE GROUND first.
  • Run the FIRST motion pass with the robot up on a box/stand so a mis-scaled command
    spins free wheels instead of launching the robot. Keep the power/e-stop in reach.
  • Ctrl-C at any time sends a zero command and disconnects.
  • Commands are streamed at ~20 Hz for their duration (the Pi host watchdog stops the
    motors if it stops receiving commands — a single send + sleep would just halt).

DEPENDS ONLY ON lerobot. The import + config field names below are version-dependent;
adjust the CONFIG block to match the lerobot you already teleop with.
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

# ============================ CONFIG — adjust to your lerobot ============================
# The Pi's LAN IP — the same address your working teleop client connects to.
DEFAULT_PI_IP = "10.0.0.125"
DEFAULT_ROBOT_ID = "lekiwi"
# The camera key the world model uses. get_observation() may expose it as
# "observation.images.top" or just "top" depending on lerobot version — we search for it.
TOP_CAMERA_HINT = "top"

# lerobot LeKiwi client import (recent lerobot). If yours differs, fix these two lines.
def _import_lekiwi():
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig  # type: ignore
    return LeKiwiClient, LeKiwiClientConfig
# Fallback import paths some lerobot versions use:
#   from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
# ========================================================================================

# Hard safety caps (do not raise without lifting the wheels and thinking it through).
VX_TEST = 0.05            # m/s forward test (dataset range is [0, 0.10] m/s) → ~5 cm/s
THETA_PROBE = 0.30        # safe in BOTH unit hypotheses (rad/s→~17°/s, deg/s→~0.3°/s)
THETA_DEG_TEST = 12.0     # only sent AFTER we conclude deg/s → ~12°/s, in-range
CMD_HZ = 20.0             # streaming rate while a command is active
DEFAULT_DURATION = 1.5    # seconds per commanded motion


def find_key(keys, *hints):
    """First key containing all hints (case-insensitive); else None."""
    for k in keys:
        kl = k.lower()
        if all(h.lower() in kl for h in hints):
            return k
    return None


def classify_base_vel_keys(action_keys):
    """Map action-feature velocity keys to x / y / theta (robust to naming)."""
    vel = [k for k in action_keys if k.lower().endswith(".vel") or ".vel" in k.lower()]
    out = {"x": None, "y": None, "theta": None}
    for k in vel:
        kl = k.lower()
        if "theta" in kl or "rot" in kl or "yaw" in kl:
            out["theta"] = k
        elif kl.split(".")[0].endswith("x") or ".x" in kl or kl.startswith("x"):
            out["x"] = k
        elif kl.split(".")[0].endswith("y") or ".y" in kl or kl.startswith("y"):
            out["y"] = k
    return out, vel


def feature_keys(robot, which):
    """Return the list of observation/action feature keys, across lerobot versions."""
    for attr in (f"{which}_features", f"{which}_feature", "features"):
        feats = getattr(robot, attr, None)
        if isinstance(feats, dict) and feats:
            return list(feats.keys())
    return None


def get_obs(robot):
    return robot.get_observation()


def build_action(obs, action_keys, base_overrides):
    """Hold every arm `.pos` at its observed value; set base `.vel` from overrides; zero the rest."""
    action = {}
    unmapped = []
    for k in action_keys:
        if k in base_overrides:
            action[k] = float(base_overrides[k])
        elif k in obs:                     # arm joints: hold current position
            action[k] = float(np.asarray(obs[k]).reshape(-1)[0])
        else:
            action[k] = 0.0
            unmapped.append(k)
    return action, unmapped


def stream(robot, obs_fn, action_keys, base_overrides, duration, label):
    """Stream `base_overrides` (+ arm hold) at CMD_HZ for `duration` s, then stop."""
    print(f"\n  → {label}: {base_overrides}  for {duration:.1f}s")
    period = 1.0 / CMD_HZ
    t_end = time.monotonic() + duration
    n = 0
    last_obs = None
    while time.monotonic() < t_end:
        obs = obs_fn(robot)
        last_obs = obs
        action, _ = build_action(obs, action_keys, base_overrides)
        robot.send_action(action)
        n += 1
        time.sleep(period)
    # cross-check: what did the robot REPORT as its base velocity while commanded?
    # (a registered-but-physically-still command vs an outright-rejected one look different here)
    if last_obs is not None:
        reported = {k: round(float(np.asarray(last_obs[k]).reshape(-1)[0]), 4)
                    for k in last_obs if isinstance(k, str) and k.lower().endswith(".vel")}
        print(f"     robot-reported base vel mid-motion: {reported}")
    # explicit stop (zero all base vels, keep arm held)
    obs = obs_fn(robot)
    zero = {k: 0.0 for k in base_overrides}
    action, _ = build_action(obs, action_keys, zero)
    robot.send_action(action)
    print(f"     ({n} commands streamed; sent stop)")


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def main():
    ap = argparse.ArgumentParser(description="LeKiwi 6b.0 transport + units smoke test")
    ap.add_argument("--ip", default=DEFAULT_PI_IP, help="Pi host LAN IP")
    ap.add_argument("--id", default=DEFAULT_ROBOT_ID, help="robot id / calibration name")
    ap.add_argument("--out", default="viz/lekiwi_6b0", help="where to save the captured top frame")
    ap.add_argument("--rtt-n", type=int, default=30, help="get_observation() calls to time")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="seconds per motion test")
    ap.add_argument("--enable-motion", action="store_true",
                    help="REQUIRED to command any base motion (else only connect/frame/RTT)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- connect ----
    try:
        LeKiwiClient, LeKiwiClientConfig = _import_lekiwi()
    except Exception as e:
        sys.exit(f"[import] could not import LeKiwiClient — fix the CONFIG import block: {e}")

    print(f"[connect] LeKiwiClient → {args.ip} (id={args.id}) …")
    # NOTE: config field names are version-dependent; `remote_ip` + `id` match recent lerobot.
    cfg = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    robot = LeKiwiClient(cfg)

    # graceful stop on Ctrl-C
    def _stop(*_):
        try:
            obs = get_obs(robot)
            ak = feature_keys(robot, "action") or []
            base, _ = classify_base_vel_keys(ak)
            zero = {v: 0.0 for v in base.values() if v}
            a, _ = build_action(obs, ak, zero)
            robot.send_action(a)
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("\n[ctrl-c] sent zero + disconnected.")
        sys.exit(130)
    signal.signal(signal.SIGINT, _stop)

    robot.connect()
    print("[connect] OK")

    # ---- contract: print observation + action feature keys ----
    obs_keys = feature_keys(robot, "observation")
    act_keys = feature_keys(robot, "action")
    obs = get_obs(robot)
    if obs_keys is None:
        obs_keys = list(obs.keys())
    if act_keys is None:
        sys.exit("[contract] could not read action_features — inspect robot API and set act_keys manually.")

    print("\n[contract] observation keys:")
    for k in obs_keys:
        v = obs.get(k, None)
        shape = getattr(v, "shape", None)
        print(f"    {k}   {tuple(shape) if shape is not None else type(v).__name__}")
    print("[contract] action keys:", act_keys)

    base_keys, vel_keys = classify_base_vel_keys(act_keys)
    print(f"[contract] base velocity keys → x={base_keys['x']}  y={base_keys['y']}  theta={base_keys['theta']}")
    if not base_keys["x"] or not base_keys["theta"]:
        print("    [warn] could not auto-map x/theta base vel keys — set them manually before motion tests.")

    # ---- capture + save the top frame ----
    top_key = None
    for k in obs_keys:
        if find_key([k], TOP_CAMERA_HINT, "image") or (TOP_CAMERA_HINT in k.lower() and "vel" not in k.lower()):
            cand = obs.get(k)
            if getattr(cand, "ndim", 0) >= 2:
                top_key = k
                break
    if top_key is None:
        print(f"\n[frame] [warn] no `{TOP_CAMERA_HINT}` image key found — is the host streaming the top camera?")
    else:
        frame = np.asarray(obs[top_key])
        print(f"\n[frame] top camera key = {top_key}  shape={frame.shape}  dtype={frame.dtype}  "
              f"min={frame.min()} max={frame.max()}")
        img = frame
        if img.ndim == 3 and img.shape[0] in (1, 3):       # CHW → HWC
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(img.squeeze()).save(out / "top_frame.png")
            print(f"[frame] saved → {out/'top_frame.png'}  (eyeball: is this the trained top view, right exposure?)")
        except Exception as e:
            np.save(out / "top_frame.npy", frame)
            print(f"[frame] PIL unavailable ({e}); saved raw array → {out/'top_frame.npy'}")

    # ---- round-trip latency ----
    print(f"\n[rtt] timing {args.rtt_n} get_observation() calls …")
    dts = []
    for _ in range(args.rtt_n):
        t0 = time.monotonic()
        get_obs(robot)
        dts.append((time.monotonic() - t0) * 1000.0)
    dts = np.array(dts)
    print(f"[rtt] get_observation: mean={dts.mean():.1f}ms  p50={np.percentile(dts,50):.1f}ms  "
          f"p95={np.percentile(dts,95):.1f}ms  (want < ~1000ms; LAN should be tens of ms)")

    # ---- motion + units (gated) ----
    if not args.enable_motion:
        print("\n[motion] skipped (no --enable-motion). Connect/frame/RTT checks done. "
              "Re-run with --enable-motion AND the wheels off the ground to test units.")
        robot.disconnect()
        return

    print("\n" + "=" * 80)
    print("MOTION TESTS — the robot will move. SAFETY: wheels OFF the ground for the first pass.")
    print("=" * 80)
    if ask("Are the wheels off the ground (robot on a stand)? [y/N] ") != "y":
        print("[motion] aborted — lift the wheels and re-run.")
        robot.disconnect()
        return

    xk, tk = base_keys["x"], base_keys["theta"]
    if not xk or not tk:
        print("[motion] aborted — base vel keys not mapped.")
        robot.disconnect()
        return

    # 1) forward sign + magnitude (x.vel assumed m/s)
    stream(robot, get_obs, act_keys, {xk: +VX_TEST}, args.duration,
           f"FORWARD test  {xk}=+{VX_TEST} (expect m/s → ~{VX_TEST*100:.0f} cm/s)")
    fwd = ask("    Which way did it drive? [f]orward (camera direction) / [b]ackward / [n]one: ")
    print(f"    → x.vel sign: {'+x = FORWARD' if fwd.startswith('f') else '+x = BACKWARD' if fwd.startswith('b') else 'NO MOTION (check mapping/units)'}")

    # 2) theta units — safe escalation
    stream(robot, get_obs, act_keys, {tk: +THETA_PROBE}, args.duration,
           f"TURN probe  {tk}=+{THETA_PROBE} (safe in both: rad/s→~17°/s, deg/s→~0.3°/s)")
    rot = ask("    Did it rotate NOTICEABLY (clearly turning)? [y/N] ")
    if rot == "y":
        native = "rad/s"
        side = ask("    Which way did the front swing? [l]eft (CCW) / [r]ight (CW): ")
        print(f"    → theta.vel native unit = RAD/S (no conversion needed). +theta = "
              f"{'LEFT/CCW' if side.startswith('l') else 'RIGHT/CW' if side.startswith('r') else '??'}")
        print("    ⇒ controller: send ω directly in rad/s.")
    else:
        print("    (no visible rotation at 0.3 → consistent with native DEG/S; confirming with a 12°/s test)")
        stream(robot, get_obs, act_keys, {tk: +THETA_DEG_TEST}, args.duration,
               f"TURN confirm  {tk}=+{THETA_DEG_TEST} (deg/s hypothesis → ~{THETA_DEG_TEST:.0f}°/s)")
        rot2 = ask("    Did it rotate noticeably now? [y/N] ")
        if rot2 == "y":
            side = ask("    Which way did the front swing? [l]eft (CCW) / [r]ight (CW): ")
            print(f"    → theta.vel native unit = DEG/S. +theta = "
                  f"{'LEFT/CCW' if side.startswith('l') else 'RIGHT/CW' if side.startswith('r') else '??'}")
            print("    ⇒ controller MUST convert ω(rad/s) → deg/s via ×180/π (≈×57.3) before send_action.")
        else:
            print("    → still no rotation — neither hypothesis produced motion. Check the theta key mapping, "
                  "host config, and that the base motors are enabled.")

    # park: zero, then disconnect
    obs = get_obs(robot)
    a, _ = build_action(obs, act_keys, {xk: 0.0, tk: 0.0, **({base_keys['y']: 0.0} if base_keys['y'] else {})})
    robot.send_action(a)
    robot.disconnect()
    print("\n[done] disconnected. Record the x.vel sign + theta unit/sign in context — they pin the "
          "(Δx,Δθ)→velocity conversion for 6b.1 and the live controller.")


if __name__ == "__main__":
    main()
