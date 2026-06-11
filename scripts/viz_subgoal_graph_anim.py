"""Animated explainer of the C3 token-space subgoal graph (GIF).

Phase 1  Collect : teleop episodes trace across the room (chunk frames).
Phase 2  Build   : frames merge into nodes (junctions green), temporal edges
                   draw per episode, cross-episode shortcut welds fade in.
Phase 3  Navigate: localize + insert goal, Dijkstra route reveals, then the
                   robot basin-hops — waypoint held 2-3 edges ahead, switching
                   forward whenever the robot closes within the switch radius.

Schematic: 2D position stands in for (position, heading) token-space nodes;
metric distances stand in for token cosine. Output: context/figures/subgoal-graph-anim.gif
"""

import heapq

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

rng = np.random.default_rng(7)

ROOM_W, ROOM_H = 3.0, 2.4
CHUNK_SPACING = 0.085
MERGE_R = 0.065
TAU = 0.14
LOOKAHEAD = 6           # edges ahead the waypoint is held
SWITCH_R = 0.18         # "close enough, advance the waypoint"
EP_COLORS = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7", "#c4ad66"]

LANDMARKS = [
    ("fan", 0.18, 0.30), ("TV + desk", 0.45, 2.22), ("chair", 1.45, 2.25),
    ("lamp", 2.55, 2.20), ("bed", 2.82, 1.20), ("hamper", 2.55, 0.22),
    ("purifier", 1.40, 0.15),
]


def smooth_path(waypoints, pts_per_seg=30):
    wp = np.asarray(waypoints)
    p = np.vstack([wp[0], wp, wp[-1]])
    out = []
    for i in range(1, len(p) - 2):
        t = np.linspace(0, 1, pts_per_seg, endpoint=False)[:, None]
        a = 2 * p[i] + (-p[i - 1] + p[i + 1]) * t
        b = (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t**2
        c = (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t**3
        out.append(0.5 * (a + b + c))
    return np.vstack(out)


def resample(path, spacing):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 2)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, path[:, 0]), np.interp(si, s, path[:, 1])])


EP_WAYPOINTS = [
    [(0.35, 0.45), (0.80, 1.40), (1.50, 1.90), (2.30, 1.85), (2.60, 1.10), (2.30, 0.45)],
    [(2.60, 0.40), (1.80, 0.80), (1.50, 1.85), (0.70, 1.95), (0.40, 1.10), (0.90, 0.40)],
    [(0.45, 1.95), (1.20, 1.30), (2.25, 1.80), (2.55, 0.90), (1.60, 0.45), (0.60, 0.70)],
    [(1.40, 0.40), (1.55, 1.20), (0.80, 1.45), (1.10, 2.00), (2.00, 1.95), (2.45, 1.40)],
    [(2.40, 1.90), (1.70, 1.55), (1.30, 0.85), (0.45, 0.85), (0.55, 1.60), (1.30, 1.95)],
]
episodes = []
for wps in EP_WAYPOINTS:
    wp = np.asarray(wps) + rng.normal(0, 0.045, (len(wps), 2))
    episodes.append(resample(smooth_path(wp), CHUNK_SPACING))

# --- graph build (same logic as the static figure) ---
nodes, node_eps, frame2node = [], [], []
for ei, ep in enumerate(episodes):
    ids = []
    for pt in ep:
        if nodes:
            d = np.linalg.norm(np.asarray(nodes) - pt, axis=1)
            j = int(np.argmin(d))
            if d[j] < MERGE_R:
                nodes[j] = 0.5 * (np.asarray(nodes[j]) + pt)
                node_eps[j].add(ei)
                ids.append(j)
                continue
        nodes.append(pt)
        node_eps.append({ei})
        ids.append(len(nodes) - 1)
    frame2node.append(ids)
nodes = np.asarray(nodes)

temporal_by_ep = []
for ids in frame2node:
    edges = [(a, b) for a, b in zip(ids[:-1], ids[1:]) if a != b]
    temporal_by_ep.append(edges)
temporal = {(min(a, b), max(a, b)) for edges in temporal_by_ep for a, b in edges}
shortcut = []
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if (i, j) in temporal or not node_eps[i].isdisjoint(node_eps[j]):
            continue
        if np.linalg.norm(nodes[i] - nodes[j]) < TAU:
            shortcut.append((i, j))

adj = {i: [] for i in range(len(nodes))}
for a, b in temporal | set(shortcut):
    w = np.linalg.norm(nodes[a] - nodes[b])
    adj[a].append((b, w))
    adj[b].append((a, w))


def dijkstra(src, dst):
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


start_pos, goal_pos = np.array([0.38, 0.42]), np.array([2.72, 1.22])
src = int(np.argmin(np.linalg.norm(nodes - start_pos, axis=1)))
dst = int(np.argmin(np.linalg.norm(nodes - goal_pos, axis=1)))
route = dijkstra(src, dst)
route_pts = nodes[route]

# robot trajectory = dense interpolation along the route polyline
seg = np.linalg.norm(np.diff(route_pts, axis=0), axis=1)
s = np.concatenate([[0], np.cumsum(seg)])
robot_path = np.column_stack([
    np.interp(np.linspace(0, s[-1], 160), s, route_pts[:, 0]),
    np.interp(np.linspace(0, s[-1], 160), s, route_pts[:, 1]),
])

# --- merge-event timeline (for the slowed-down Phase 2a) -----------------
# Re-simulate the build frame-by-frame: after each ingested raw frame, which
# nodes exist and how many episodes have merged into each.
merge_events = []           # per raw frame: (node_id, n_episodes_after)
_eps_so_far = {}
for ei, ids in enumerate(frame2node):
    for n in ids:
        _eps_so_far.setdefault(n, set()).add(ei)
        merge_events.append((n, len(_eps_so_far[n])))
N_RAW = len(merge_events)

# --- frame schedule ------------------------------------------------------
F_PER_EP = 22           # frames to trace one episode
F_MERGE = 56            # 2a: raw frames ingest one by one, nodes mint/merge
F_MERGE_H = 22          # hold: what the merge produced
F_TEMP = 40             # 2b: temporal edges draw episode by episode
F_TEMP_H = 22           # hold: the certified skeleton
F_SHORT = 44            # 2c: shortcut welds fade in
F_SHORT_H = 26          # hold: one mesh
F_LOCAL = 16            # start/goal markers
F_ROUTE = 24            # route reveal
F_DRIVE = len(robot_path)
F_HOLD = 18             # end hold
PH1 = len(episodes) * F_PER_EP
PH2 = PH1 + F_MERGE + F_MERGE_H + F_TEMP + F_TEMP_H + F_SHORT + F_SHORT_H
PH3 = PH2 + F_LOCAL + F_ROUTE + F_DRIVE + F_HOLD
TOTAL = PH3

fig, ax = plt.subplots(figsize=(8.4, 7.0))
fig.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.97)


def base_room(dim_eps=0.0):
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, fc="#f7f4ee", ec="#444", lw=1.5, zorder=0))
    ax.add_patch(Rectangle((0.75, 0.55), 1.5, 1.3, fc="#efe9dd", ec="none", zorder=0))
    for name, x, y in LANDMARKS:
        ax.plot(x, y, "s", color="#555", ms=8, zorder=3)
        dy = 0.09 if y < 1.2 else -0.14
        ax.text(x, y + dy, name, fontsize=9, ha="center", color="#333", zorder=3)
    ax.set_xlim(-0.12, ROOM_W + 0.12)
    ax.set_ylim(-0.12, ROOM_H + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")


def header(phase, caption):
    fig.suptitle(phase, fontsize=14, fontweight="bold", y=0.975)
    if len(caption) > 95 and "\n" not in caption:   # wrap long captions near the middle
        words = caption.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) > 92:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        caption = "\n".join(lines)
    fig.text(0.5, 0.925, caption, fontsize=9.5, ha="center", va="top", color="#444")


def draw_graph(node_alpha=1.0, edge_alpha=1.0, short_frac=1.0, dim=False):
    c_t = "#d8dee6" if dim else "#9ab2d0"
    c_s = "#f0d3ae" if dim else "#e07b00"
    for a, b in temporal:
        ax.plot(*zip(nodes[a], nodes[b]), "-", color=c_t, lw=1.3, alpha=edge_alpha, zorder=2)
    ns = shortcut[: int(len(shortcut) * short_frac)]
    for a, b in ns:
        ax.plot(*zip(nodes[a], nodes[b]), "--", color=c_s, lw=1.0, alpha=edge_alpha, zorder=2)
    junction = [i for i, eps in enumerate(node_eps) if len(eps) >= 2]
    plain = [i for i, eps in enumerate(node_eps) if len(eps) == 1]
    cp = "#c9d3de" if dim else "#3b6ea5"
    cj = "#bcd9c5" if dim else "#1a9850"
    ax.plot(nodes[plain, 0], nodes[plain, 1], "o", color=cp, ms=3.6, alpha=node_alpha, zorder=3)
    ax.plot(nodes[junction, 0], nodes[junction, 1], "o", color=cj, ms=6.5,
            mec="k" if not dim else "none", mew=0.6, alpha=node_alpha, zorder=4)


def draw_frame(f):
    ax.clear()
    for txt in list(fig.texts):
        txt.remove()

    # ---------------- Phase 1: collect ----------------
    if f < PH1:
        base_room()
        header("Phase 1 · Collect — teleop episodes",
               "each dot = one chunk-boundary frame (a real photo + pose) · threads crisscross the room")
        ei_now, prog = divmod(f, F_PER_EP)
        for ei in range(ei_now + 1):
            ep = episodes[ei]
            n = len(ep) if ei < ei_now else max(2, int(len(ep) * (prog + 1) / F_PER_EP))
            ax.plot(ep[:n, 0], ep[:n, 1], "-", color=EP_COLORS[ei], lw=1.5, alpha=0.8, zorder=2)
            ax.plot(ep[:n, 0], ep[:n, 1], "o", color=EP_COLORS[ei], ms=3.4, zorder=2)
            ax.plot(*ep[0], "o", color=EP_COLORS[ei], ms=9, mec="k", mew=0.8, zorder=4)
            if ei == ei_now:
                ax.plot(*ep[n - 1], "o", color=EP_COLORS[ei], ms=10, mec="k", mew=1.2, zorder=5)
        ax.text(0.02, 0.02, f"episode {ei_now + 1}/{len(episodes)}",
                transform=ax.transAxes, fontsize=10, color="#555")
        return

    # ---------------- Phase 2: build ----------------
    if f < PH2:
        g = f - PH1
        base_room()
        for ei, ep in enumerate(episodes):   # dim the raw threads underneath
            ax.plot(ep[:, 0], ep[:, 1], "-", color=EP_COLORS[ei], lw=1.0, alpha=0.12, zorder=1)

        def nodes_at(n_proc, ring_window=0):
            """Node states after ingesting n_proc raw frames (+ recent-change rings)."""
            eps_count, recent = {}, set()
            for fi, (nid, cnt) in enumerate(merge_events[:n_proc]):
                eps_count[nid] = cnt
                if fi >= n_proc - ring_window:
                    recent.add(nid)
            plain = [n for n, c in eps_count.items() if c == 1]
            junc = [n for n, c in eps_count.items() if c >= 2]
            if plain:
                ax.plot(nodes[plain, 0], nodes[plain, 1], "o", color="#3b6ea5", ms=3.6, zorder=3)
            if junc:
                ax.plot(nodes[junc, 0], nodes[junc, 1], "o", color="#1a9850",
                        ms=6.5, mec="k", mew=0.6, zorder=4)
            for n in recent:
                ax.add_patch(Circle(nodes[n], 0.075, fc="none", ec="#e07b00", lw=1.6, zorder=5))
            return len(eps_count), len(junc)

        if g < F_MERGE:                      # ---- 2a: ingest + merge, frame by frame
            header("Phase 2a · MERGE — same view ⇒ same node",
                   "ingest frames one by one: within the noise floor of an existing node → absorbed; "
                   "else mint a new node · green = a later episode merged in (junction)")
            n_proc = max(2, int(N_RAW * (g + 1) / F_MERGE))
            n_nodes, n_junc = nodes_at(n_proc, ring_window=max(4, N_RAW // F_MERGE))
            cur = merge_events[n_proc - 1][0]
            ax.plot(*nodes[cur], "+", color="k", ms=13, mew=2, zorder=6)
            ax.text(0.02, 0.02,
                    f"frames ingested: {n_proc}/{N_RAW} → nodes: {n_nodes} (junctions: {n_junc})",
                    transform=ax.transAxes, fontsize=10, color="#555")
            return
        g -= F_MERGE
        if g < F_MERGE_H:                    # ---- hold after 2a
            header("Phase 2a · MERGE — what we now have",
                   "every node keeps one REAL representative frame (never an average — it must serve as a "
                   "CEM goal image) · junctions are where 5 separate threads became shared waypoints")
            nodes_at(N_RAW)
            ax.text(0.02, 0.02,
                    f"{N_RAW} frames → {len(nodes)} pose-nodes · same spot at a different heading does "
                    f"NOT merge (yaw is a different view)",
                    transform=ax.transAxes, fontsize=9.5, color="#555")
            return
        g -= F_MERGE_H
        if g < F_TEMP:                       # ---- 2b: temporal edges, episode by episode
            header("Phase 2b · TEMPORAL EDGES — certified by execution",
                   "consecutive chunks of an episode: the robot DEMONSTRABLY drove A→B · weight = 1 chunk · "
                   "comes from TIME, not appearance — valid even where the metric is wrong")
            ep_show = max(1, int(len(temporal_by_ep) * (g + 1) / F_TEMP))
            for ei, edges in enumerate(temporal_by_ep[:ep_show]):
                lw = 2.2 if ei == ep_show - 1 else 1.4
                for a, b in edges:
                    ax.plot(*zip(nodes[a], nodes[b]), "-", color="#9ab2d0", lw=lw, zorder=2)
            nodes_at(N_RAW)
            ax.text(0.02, 0.02, f"episode {ep_show}/{len(temporal_by_ep)} chained into the graph",
                    transform=ax.transAxes, fontsize=10, color="#555")
            return
        g -= F_TEMP
        if g < F_TEMP_H:                     # ---- hold after 2b
            header("Phase 2b · TEMPORAL EDGES — the trusted skeleton",
                   "each episode is now a certified path over nodes — but the threads only interconnect "
                   "where merges happened · routes are still mostly trapped on their own episode")
            draw_graph(short_frac=0.0)
            ax.text(0.02, 0.02,
                    f"{len(temporal)} temporal edges · trust level: executed, not inferred",
                    transform=ax.transAxes, fontsize=10, color="#555")
            return
        g -= F_TEMP_H
        if g < F_SHORT:                      # ---- 2c: shortcut welds
            header("Phase 2c · SHORTCUT WELDS — inferred transfer ramps, policed",
                   "cross-episode node pairs at cosine < τ (≈ one CEM reach): 'one plan apart, not the same "
                   "pose' · the ONLY inferred edges → conservative τ + offline audit + deleted on failed hops")
            frac = (g + 1) / F_SHORT
            draw_graph(short_frac=frac)
            ax.text(0.02, 0.02,
                    f"shortcut edges: {int(len(shortcut) * frac)}/{len(shortcut)} · each claims "
                    f"'CEM can cross this gap in one plan' — testable, deletable",
                    transform=ax.transAxes, fontsize=9.5, color="#b35e00")
            return
        g -= F_SHORT
        # ---- hold after 2c
        header("Phase 2 · BUILD COMPLETE — 5 threads are now one mesh",
               "merge gave shared waypoints · temporal edges gave certified motion · shortcuts gave "
               "transfer ramps — Dijkstra can now route across episodes it never saw together")
        draw_graph()
        ax.text(0.02, 0.02,
                f"{len(nodes)} nodes · {len(temporal)} certified + {len(shortcut)} inferred edges · "
                f"zero training",
                transform=ax.transAxes, fontsize=10, color="#555")
        return

    # ---------------- Phase 3: navigate ----------------
    g = f - PH2
    base_room()
    draw_graph(dim=True)

    def start_goal():
        ax.plot(*nodes[src], "o", color="k", ms=12, zorder=6)
        ax.text(*(nodes[src] + [0.04, -0.17]), "START", fontsize=9, fontweight="bold", zorder=6)
        ax.plot(*nodes[dst], "*", color="#d62728", ms=22, mec="k", zorder=6)
        ax.text(*(nodes[dst] + [-0.28, 0.13]), "GOAL", fontsize=9, fontweight="bold",
                color="#d62728", zorder=6)

    if g < F_LOCAL:
        header("Phase 3 · Navigate — localize + insert the goal",
               "k-NN in token space: current frame → nearest node · goal image → nearest node (far NN ⇒ out of coverage, refuse)")
        start_goal()
        r = 0.30 * (1 - g / F_LOCAL) + 0.06
        ax.add_patch(Circle(nodes[src], r, fc="none", ec="k", lw=1.5, zorder=5))
        ax.add_patch(Circle(nodes[dst], r, fc="none", ec="#d62728", lw=1.5, zorder=5))
        return

    if g < F_LOCAL + F_ROUTE:
        header("Phase 3 · Navigate — Dijkstra over certified hops",
               "shortest chain of short trusted edges · total weight ≈ honest chunks-to-drive · no far-field metric judgment anywhere")
        k = g - F_LOCAL
        n = max(2, int(len(route_pts) * (k + 1) / F_ROUTE))
        ax.plot(route_pts[:n, 0], route_pts[:n, 1], "-", color="#2166ac", lw=3.2, zorder=5)
        ax.plot(route_pts[:n, 0], route_pts[:n, 1], "o", color="#2166ac", ms=5.5, zorder=5)
        start_goal()
        return

    if g < F_LOCAL + F_ROUTE + F_DRIVE:
        k = g - F_LOCAL - F_ROUTE
        pos = robot_path[k]
        # waypoint: held LOOKAHEAD edges past the nearest route node, advanced by switch radius
        near = int(np.argmin(np.linalg.norm(route_pts - pos, axis=1)))
        wp_i = min(near + LOOKAHEAD, len(route) - 1)
        while wp_i < len(route) - 1 and np.linalg.norm(nodes[route[wp_i]] - pos) < SWITCH_R:
            wp_i += 1
        wp = nodes[route[wp_i]]
        header("Phase 3 · Navigate — basin-hopping",
               "CEM target = real frame 2–3 edges ahead (sharp basin, the 3/3 regime) · "
               "switch at d < ~0.2 → re-localize → advance")
        ax.plot(route_pts[:, 0], route_pts[:, 1], "-", color="#2166ac", lw=2.0, alpha=0.35, zorder=4)
        ax.plot(robot_path[: k + 1, 0], robot_path[: k + 1, 1], "-", color="#1a9850",
                lw=3.0, zorder=5)
        ax.add_patch(Circle(wp, 0.20, fc="#2166ac", alpha=0.15, ec="#2166ac", lw=1.8, zorder=4))
        ax.plot(*wp, "o", color="#2166ac", ms=11, mec="k", mew=1.0, zorder=6)
        ax.text(*(wp + [0.05, 0.10]), "waypoint", fontsize=9, color="#2166ac",
                fontweight="bold", zorder=6)
        ax.plot(*pos, "o", color="#1a9850", ms=13, mec="k", mew=1.4, zorder=7)
        ax.text(*(pos + [0.05, -0.16]), "robot", fontsize=9, color="#1a9850",
                fontweight="bold", zorder=7)
        start_goal()
        d_now = np.linalg.norm(pos - nodes[dst])
        ax.text(0.02, 0.02, f"hop target: {wp_i}/{len(route) - 1} · CEM sees only the shaded basin",
                transform=ax.transAxes, fontsize=10, color="#1a9850")
        return

    # end hold
    header("Phase 3 · Navigate — arrived",
           "every subgoal was a REAL dataset frame · the WM only ever rolled out ~3 chunks from a real pose")
    ax.plot(route_pts[:, 0], route_pts[:, 1], "-", color="#2166ac", lw=2.0, alpha=0.35, zorder=4)
    ax.plot(robot_path[:, 0], robot_path[:, 1], "-", color="#1a9850", lw=3.0, zorder=5)
    start_goal()
    ax.plot(*robot_path[-1], "o", color="#1a9850", ms=13, mec="k", mew=1.4, zorder=7)
    ax.text(0.02, 0.02, "reach-thresh fires on the FINAL node only",
            transform=ax.transAxes, fontsize=10, color="#1a9850")


FPS = 8   # slow playback; mp4 is pause/scrub/speed-controllable in any player
anim = FuncAnimation(fig, draw_frame, frames=TOTAL, interval=1000 / FPS)
if "--gif" in sys.argv:
    out = "context/figures/subgoal-graph-anim.gif"
    anim.save(out, writer=PillowWriter(fps=14), dpi=92)
else:
    out = "context/figures/subgoal-graph-anim.mp4"
    anim.save(out, writer=FFMpegWriter(fps=FPS, bitrate=2400,
                                       extra_args=["-pix_fmt", "yuv420p"]), dpi=130)
print(f"saved {out}  ({TOTAL} frames @{FPS}fps ≈ {TOTAL / FPS:.0f}s)")
