"""Three-panel explainer of the C3 token-space subgoal graph.

Panel 1: raw teleop episodes = threads of chunk-boundary frames.
Panel 2: graph build — merge near-identical frames into nodes (junctions),
         temporal edges (certified), shortcut edges (inferred welds).
Panel 3: runtime — localize, Dijkstra route, hand CEM the waypoint 2-3 edges
         ahead, basin-hop to the goal.

Schematic only: 2D positions stand in for (position, heading) token-space
nodes; distances stand in for token cosine. Output: context/figures/subgoal-graph-viz.png
"""

import heapq

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

rng = np.random.default_rng(7)

ROOM_W, ROOM_H = 3.0, 2.4
CHUNK_SPACING = 0.085   # m between chunk-boundary frames (schematic)
MERGE_R = 0.065         # "same pose" -> merge into one node (~noise floor)
TAU = 0.14              # shortcut admission radius (~one CEM reach)
EP_COLORS = ["#4878cf", "#6acc65", "#d65f5f", "#b47cc7", "#c4ad66", "#77bedb"]

LANDMARKS = [
    ("fan", 0.18, 0.30), ("TV + desk", 0.45, 2.22), ("chair", 1.45, 2.25),
    ("lamp", 2.55, 2.20), ("bed", 2.82, 1.20), ("hamper", 2.55, 0.22),
    ("purifier", 1.40, 0.15),
]


def smooth_path(waypoints, pts_per_seg=30):
    """Catmull-Rom through waypoints -> dense smooth path."""
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


# --- 1. synthesize loopy teleop episodes (waypoints hand-placed to crisscross) ---
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

# --- 2. build the graph: greedy merge -> nodes; temporal + shortcut edges ---
nodes, node_eps = [], []          # node positions; set of episodes merged into it
frame2node = []                   # per episode: frame index -> node id
for ei, ep in enumerate(episodes):
    ids = []
    for pt in ep:
        if nodes:
            d = np.linalg.norm(np.asarray(nodes) - pt, axis=1)
            j = int(np.argmin(d))
            if d[j] < MERGE_R:                      # same pose -> MERGE
                nodes[j] = 0.5 * (np.asarray(nodes[j]) + pt)  # running average
                node_eps[j].add(ei)
                ids.append(j)
                continue
        nodes.append(pt)
        node_eps.append({ei})
        ids.append(len(nodes) - 1)
    frame2node.append(ids)
nodes = np.asarray(nodes)

temporal, shortcut = set(), set()
for ids in frame2node:
    for a, b in zip(ids[:-1], ids[1:]):
        if a != b:
            temporal.add((min(a, b), max(a, b)))
junctions = {n for ids in frame2node for n in ids} & {
    n for i, ids in enumerate(frame2node) for n in ids
    if any(n in o for j, o in enumerate(frame2node) if j != i)
}
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if (i, j) in temporal or not node_eps[i].isdisjoint(node_eps[j]):
            continue  # only cross-episode welds — within-episode is temporal's job
        if np.linalg.norm(nodes[i] - nodes[j]) < TAU:   # close pose -> CONNECT
            shortcut.add((i, j))

adj = {i: [] for i in range(len(nodes))}
for a, b in temporal | shortcut:
    w = np.linalg.norm(nodes[a] - nodes[b])
    adj[a].append((b, w))
    adj[b].append((a, w))


def dijkstra(src, dst):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
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


start_pos, goal_pos = np.array([0.38, 0.42]), np.array([2.72, 1.22])  # fan -> bed
src = int(np.argmin(np.linalg.norm(nodes - start_pos, axis=1)))
dst = int(np.argmin(np.linalg.norm(nodes - goal_pos, axis=1)))
route = dijkstra(src, dst)

# --- 3. draw -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.4))
fig.suptitle("C3 token-space subgoal graph — from teleop threads to basin-hopping",
             fontsize=15, fontweight="bold", y=0.99)


def room(ax, title, subtitle):
    ax.add_patch(Rectangle((0, 0), ROOM_W, ROOM_H, fc="#f7f4ee", ec="#444", lw=1.5, zorder=0))
    ax.add_patch(Rectangle((0.75, 0.55), 1.5, 1.3, fc="#efe9dd", ec="none", zorder=0))
    ax.text(1.5, 1.2, "low-texture rug centre", color="#b9ac94", fontsize=8,
            ha="center", style="italic", zorder=1)
    for name, x, y in LANDMARKS:
        ax.plot(x, y, "s", color="#555", ms=7, zorder=3)
        dy = 0.09 if y < 1.2 else -0.13
        ax.text(x, y + dy, name, fontsize=8, ha="center", color="#333", zorder=3)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.5, 1.075, subtitle, transform=ax.transAxes, fontsize=8.5,
            ha="center", color="#555")
    ax.set_xlim(-0.15, ROOM_W + 0.15)
    ax.set_ylim(-0.15, ROOM_H + 0.28)
    ax.set_aspect("equal")
    ax.axis("off")


# Panel 1 — raw episodes
ax = axes[0]
room(ax, "1 · Raw data: 50 teleop episodes",
     "each dot = one chunk-boundary frame (a real photo + pose); threads crisscross the room")
for ei, ep in enumerate(episodes):
    ax.plot(ep[:, 0], ep[:, 1], "-", color=EP_COLORS[ei], lw=1.4, alpha=0.75, zorder=2)
    ax.plot(ep[:, 0], ep[:, 1], "o", color=EP_COLORS[ei], ms=3.2, alpha=0.85, zorder=2)
    ax.plot(*ep[0], "o", color=EP_COLORS[ei], ms=8, mec="k", mew=0.8, zorder=4)
crossing = np.array([1.50, 1.87])
ax.add_patch(Circle(crossing, 0.17, fc="none", ec="#e07b00", lw=2, ls="--", zorder=5))
ax.annotate("different episodes pass\nthrough the same place",
            xy=crossing + [0.1, -0.14], xytext=(1.95, 0.95), fontsize=8.5, color="#e07b00",
            arrowprops=dict(arrowstyle="->", color="#e07b00"), zorder=6)

# Panel 2 — graph build
ax = axes[1]
room(ax, "2 · Build: merge → nodes, two kinds of edges",
     f"frames < merge-radius collapse to ONE node (junction) · temporal edges certified · "
     f"shortcuts admitted at cosine < τ")
for a, b in temporal:
    ax.plot(*zip(nodes[a], nodes[b]), "-", color="#9ab2d0", lw=1.5, zorder=2)
for a, b in shortcut:
    ax.plot(*zip(nodes[a], nodes[b]), "--", color="#e07b00", lw=1.1, alpha=0.85, zorder=2)
junction = [i for i, eps in enumerate(node_eps) if len(eps) >= 2]
plain = [i for i, eps in enumerate(node_eps) if len(eps) == 1]
ax.plot(nodes[plain, 0], nodes[plain, 1], "o", color="#3b6ea5", ms=4.5, zorder=3)
if junction:
    jm = nodes[junction]
    ax.plot(jm[:, 0], jm[:, 1], "o", color="#1a9850", ms=7.5, mec="k", mew=0.7, zorder=4)
ax.plot([], [], "o", color="#3b6ea5", ms=5, label="node (merged frames)")
ax.plot([], [], "o", color="#1a9850", ms=7.5, mec="k", label="junction (multi-episode merge)")
ax.plot([], [], "-", color="#9ab2d0", lw=1.5, label="temporal edge (robot drove it)")
ax.plot([], [], "--", color="#e07b00", lw=1.2, label="shortcut edge (cosine < τ)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), fontsize=8, ncol=2, frameon=False)

# Panel 3 — runtime
ax = axes[2]
room(ax, "3 · Runtime: localize → Dijkstra → hand CEM one basin at a time",
     "waypoint = real frame 2–3 edges ahead; switch at d < ~0.2; only the FINAL hop uses reach-thresh")
for a, b in temporal:
    ax.plot(*zip(nodes[a], nodes[b]), "-", color="#d8dee6", lw=1.0, zorder=1)
for a, b in shortcut:
    ax.plot(*zip(nodes[a], nodes[b]), "--", color="#f0d3ae", lw=0.9, zorder=1)
ax.plot(nodes[:, 0], nodes[:, 1], "o", color="#c9d3de", ms=3.5, zorder=2)
rp = nodes[route]
ax.plot(rp[:, 0], rp[:, 1], "-", color="#2166ac", lw=3.2, alpha=0.9, zorder=4)
ax.plot(rp[:, 0], rp[:, 1], "o", color="#2166ac", ms=6, zorder=5)
ax.plot(*nodes[src], "o", color="k", ms=11, zorder=6)
ax.text(*(nodes[src] + [0.0, -0.16]), "START\n(localized by k-NN)", fontsize=8,
        ha="center", fontweight="bold", zorder=6)
ax.plot(*nodes[dst], "*", color="#d62728", ms=20, mec="k", zorder=6)
ax.text(*(nodes[dst] + [-0.05, 0.14]), "GOAL image\n(inserted by k-NN)", fontsize=8,
        ha="center", fontweight="bold", color="#d62728", zorder=6)
wp_i = min(6, len(route) - 1)
wp = nodes[route[wp_i]]
ax.add_patch(Circle(wp, 0.20, fc="#2166ac", alpha=0.13, ec="#2166ac", lw=1.6, zorder=3))
ax.annotate("CEM target: waypoint 2–3 edges ahead\n(sharp basin — the regime that went 3/3)",
            xy=wp, xytext=(0.18, 1.45), fontsize=8.5, color="#2166ac", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.4), zorder=7)
arrow = FancyArrowPatch(nodes[src], wp, connectionstyle="arc3,rad=-0.25",
                        arrowstyle="-|>", mutation_scale=16, color="#1a9850", lw=2.4, zorder=6)
ax.add_patch(arrow)
ax.text(0.78, 0.18, "CEM+WM drives one basin hop,\nthen re-localize → advance waypoint",
        fontsize=8.5, color="#1a9850", fontweight="bold", zorder=7)

fig.tight_layout(rect=[0, 0, 1, 0.965])
out = "context/figures/subgoal-graph-viz.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"saved {out}  |  nodes={len(nodes)}  temporal={len(temporal)}  "
      f"shortcut={len(shortcut)}  route_hops={len(route)-1}")
