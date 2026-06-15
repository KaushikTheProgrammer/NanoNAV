# NanoNAV Technical Report — Narrative & Content Plan

Working document for the medium/substack-style write-up (likely a standalone site).
This nails down the STORY first; design/layout/platform come after.
Source of truth for every claim: `context/experiment-log.md` (chronology),
`context/roadmap.md` (stages), `context/learned-distance-metric.md`,
`context/semantic-wm-retrain.md`.

---

## Title candidates

1. **"Teaching a $300 Robot to Navigate by Imagination"** — accessible, leads with the hook
2. **"NanoNAV: Goal-Image Navigation with a World Model, a Distance Metric, and a Graph"** — descriptive
3. **"The Objective Was the Bottleneck"** — the thesis as title (essay-ish, good for substack)
4. **"From Pixels to Plans: Building Visual Navigation on a LeKiwi"** — safe/middle

Recommendation: #1 as the page title, #3 energy as the running thesis.

## The one-paragraph pitch (what the post is)

We taught a LeKiwi mobile robot to drive to any place in a room you show it a photo of —
no maps, no SLAM, no GPS, no reward function, no task demonstrations. A 160M-parameter
world model imagines what the camera would see under candidate actions; a planner (CEM)
picks the actions whose imagined future best matches the goal photo; a topological graph
built from the robot's own memories routes it across the room when the goal is too far to
"see." The interesting part isn't that it works — it's *what kept it from working*: the
journey is three confident wrong diagnoses, one measurement that turned the bottleneck into
a number, a pivot from VAE latents to semantic (DINOv2) tokens, and a graph that had to
learn the robot can't drive backwards. Total data: **50 teleop episodes (~25 minutes of
driving)**. Total model: small enough to train overnight on one GPU.

## Audience & tone

- Primary: ML/robotics-curious engineers (Medium/HN/LinkedIn crowd). Secondary: researchers
  who will skim for the Gate A table and the C0 probe matrix.
- Tone: first-person build log with real numbers. The strongest material is the
  *epistemics* — wrong hypotheses stated plainly, then the controlled experiment that
  killed each one. Don't sand that off; it's the differentiator vs. typical "look it works" posts.
- Every section ends on the question that drives the next one (the log naturally has this shape).

---

# The narrative in five acts

## Act 0 — The premise (setup, short)

**Beat:** What if a robot could navigate the way you imagine walking through your own home?
Show goal photo → robot drives there. The "world model + planner" recipe in one diagram.

Content:
- The task: goal-image navigation. Input = current camera frame + a photo of where to be.
  Output = wheel velocities. Nothing else.
- The recipe (Pattern A): action-conditioned world model (NanoWM, diffusion-forcing,
  160M) predicts future camera frames given actions; CEM samples action sequences, scores
  imagined endpoints against the goal, executes the best first step, replans (~7 s/cycle,
  stop-and-plan).
- The constraint that makes it interesting: **50 episodes / 44,926 frames / one room** —
  tiny data, one consumer GPU (RunPod H100, hours not weeks), a hobby-grade robot
  (LeKiwi: 3-omniwheel base + arm + webcams, Raspberry Pi host).
- Design decisions worth one paragraph each (with the why):
  - Action = body-frame **(Δx, Δθ)** integrated over a chunk — heading-invariant, 2-D
    (CEM searches 6 dims at horizon 3, not 30). Δy provably negligible (max 0.58 mm/chunk, measured).
  - Camera = elevated third-person (~55°): four depth zones, robot body in frame as
    ego-reference.
  - Data = *exploratory driving, not demonstrations*. Tasks enter only at inference via the
    goal image. Suboptimal data is good data — CEM needs to evaluate bad actions too.
- Figure: room photo (`context/figures/room.jpg`) + system diagram (NEW — to make).
- Fun ops aside (sidebar): `theta.vel` turned out to be **degrees/sec** — integrate as rad/s
  and one episode "spins" 7,528° (21 silent rotations). Units bugs are navigation bugs.

**Closing question:** can a 160M model trained on 25 minutes of driving learn this room's physics?

## Act I — Getting a world model that feels actions (the training saga)

**Beat:** the first model ignored its own actions, and the first two explanations were wrong.

Content:
1. **Run 001 fails the action test.** Train, run the GT-vs-zero-vs-random action diagnostic:
   the model predicts the same future no matter what you tell it the robot did
   (action-embed RMS 0.0088; zero ≈ random). A world model that ignores actions is a
   screensaver, not a simulator.
2. **Wrong diagnosis #1: "translation is invisible to this camera."** The f-sweep showed
   corr(|Δx|, latent change) ≈ 0 at every frame interval → wrote off the camera geometry.
   **The refutation is the lesson:** the correlation was the wrong estimator (bang-bang
   speeds + rotation chunks polluting it). A *controlled contrast* — stationary vs
   pure-translation chunks — showed AUC 0.94–0.98 and a clean dose-response with f.
   Translation was in the latent all along. *Lesson: a pooled correlation can hide a
   perfectly detectable signal; design the controlled test first.*
3. **Run 002 (f=10) — alive.** Clean gt < zero < random separation; decoded rollouts
   visibly track translation/rotation/arcs.
4. **val_loss lies for diffusion-forcing.** Checkpoint quality is U-shaped in rollout
   space: val-best (step 4125) is NOT the best rollout model; ~8K peaks; 12K overfits.
   Judged by rollouts, carried step-8000. *Lesson: judge world models by what they're for
   (rollouts), not the denoising proxy.*
5. **Offline planner validation (6a):** CEM hits the WM ceiling (reached_ratio ~1.0 in
   every motion bucket), recovers true commands (sign 100%, ~1 cm / ~2.5°), and the cheap
   DDIM=3 sampler holds → ~7 s replans are viable. The planner engine was never the problem
   — remember this; it becomes the running theme.
- Figures: diagnostic plot (GT/zero/random), stationary-vs-translation table or AUC chart,
  U-shape rollout-quality curve, a 6a montage (imagined rollout landing on goal).

**Closing question:** it plans perfectly offline. What happens on a real robot?

## Act II — The robot says no (closed-loop and the three wrong root causes)

**Beat:** the heart of the story. Offline-perfect, on-robot it wanders. Every "root cause"
we found was real-but-not-it, until measurement replaced intuition.

Content (keep the honest chronology — it reads like a detective story):
1. **Bring-up grit (compressed, sidebar-worthy):** units contract pinned on hardware
   (m/s forward; deg/s CCW; a low-speed rotation deadband), wheels-up testing can't show
   body rotation (omniwheels!), open-loop replay matches dead-reckoning ~0 cm through a
   117° arc.
2. **First closed-loop run: planning works, robot doesn't converge.** dist hovers ~45 for
   22 steps; theta flip-flops; robot wiggles instead of committing.
3. **Real bug #1 (necessary, not sufficient):** the live preprocess fed the VAE [0,1]
   pixels; training used [-1,1]. Fixed — still no convergence.
4. **Wrong diagnosis #2: "the wide-angle camera makes a flat objective."** A
   drive-straight probe showed dist flat for 46 cm then snapping down after a heading
   nudge → blamed camera FOV/parallax/distortion, wrote a whole theory ("camera ⊗
   objective conditioning"). **Refutation:** a controlled hand-placed radial sweep —
   the latent distance is *monotone* over 40 cm with SNR ~17σ/10 cm. The "flat 46 cm" was
   the robot drifting off-axis (path length ≠ radial approach). *Retracted in the log the
   same day. Lesson #2: measure the landscape, don't infer it from a trajectory.*
5. **First convergence!** Re-captured the goal at the true pose → REACHED ×2 in 10–14
   steps. The whole stack vindicated... near goals.
6. **But far goals still stall, and no knob helps.** Horizon 5, wider sampling variance,
   higher speed cap — nothing. The decisive observation (operator watching the robot):
   CEM *correctly turned the robot to face the chair* and the distance metric **barely
   rewarded it**. The search was choosing the right behavior; the objective couldn't see it.
7. **Then it got worse — the hallucination.** From an under-covered start pose, the WM's
   imagined rollout showed *a different part of the room* (vivid, confident, wrong).
   Diffusion models off-distribution don't degrade gracefully — they snap to a familiar
   mode. Both the cost and what CEM optimizes were garbage from those poses.
   (Figure exists: `context/figures/live-distribution-gap_*.png`.)
- The state at the bottom of the arc: **two coupled blockers** — (a) the objective is
  blind far from the goal; (b) the WM hallucinates off-coverage. Cheap fixes exhausted.

**Closing question:** stop guessing. Can we make "the objective is blind" a *number*?

## Act III — The measurement and the pivot (Gate A → DINOv2 retrain)

**Beat:** one afternoon of tape-measure robotics turned the bottleneck into a table, and
the table dictated the architecture.

Content:
1. **Gate A — the distance-metric bake-off on real frames.** Hand-place the robot at
   measured displacements (radial 10–60 cm, lateral ±60 cm, yaw ±30°), capture frames,
   grade every candidate distance on the same rig: Spearman ρ, far-band slope vs
   same-pose noise (3σ gate), yaw basin.

   | candidate | radial ρ | far-band slope/σ (radial/lateral) | verdict |
   |---|---|---|---|
   | pixel L1 | 1.00 | 706 / 386 | fail (lateral ordering) |
   | **SD-VAE latent L2 (the objective we'd been planning with)** | 1.00 | **1.25 / 0.80** | **FAIL** |
   | **frozen DINOv2 patch cosine** | 0.943 | **12 / 21** | **PASS** |

   The punchline writes itself: our planning objective was perfectly *ordered* (ρ=1.0)
   but its far-field gradient was **below the robot's own standing-still noise floor**.
   CEM literally could not see progress more than ~3 chunks out. Meanwhile a frozen,
   never-trained DINOv2 had 12–21σ of gradient *on the same pixels*. The information was
   always in the images; the representation buried it.
2. **The weld test (validate-first paid off):** WM-imagined latents sat +23σ off the
   real-frame curve, and distance *increased* within rollouts while the robot nominally
   approached. So you can't just bolt a better metric onto the old WM's imagination.
3. **Literature interlude (short, links):** ViNG/GNM/ViNT (distance head + graph is the
   field recipe), DINO-WM and RAE-NWM (planning on DINO patch tokens works; RAE-NWM beats
   the VAE-latent equivalent 79% vs 43% on Habitat), and the QRL detour we *didn't* take
   (OGBench: ~0% on visual tasks). Frame it as: the literature confirmed the shape, our
   sweep measured the thing no paper had (0–60 cm flatness at robot scale).
4. **The pivot — Option C:** don't distill DINO into the old stack; **retrain the WM to
   predict frozen DINOv2 patch tokens** so the rollout space IS the validated distance
   space, and the planning cost (token cosine) needs zero training.
5. **The Finding-#4 subplot resolves (the most "science" moment in the project).**
   Received wisdom (NanoWM's own Table 5/6): semantic latents kill action conditioning.
   The C0 probe matrix — 4 runs × 3k steps, one variable at a time — showed it was never
   the latents, it was the **conditioning path**:

   | probe | objective + injection | action RMS | verdict |
   |---|---|---|---|
   | C0d | flow + additive (the old setting) | **0.0028 — atrophy reproduced on demand** | FAIL |
   | C0b | flow + cross-attention | 0.029 | FAIL |
   | C0a | flow + AdaLN-fuse | 0.207 | PASS |
   | **C0c** | **x0 + AdaLN-fuse** | **0.182, best margin (21.3)** | **WINNER** |

   Same data, same latents — additive injection starves, AdaLN thrives. A community
   "finding" reproduced, narrowed, and explained in 12 GPU-hours.
6. **Gate C @12k:** action branch *strengthening* (RMS 0.333, margin 43.4), CEM at the WM
   ceiling again, weld ρ 0.29 → 0.876, and **the hallucination is fixed at the source**:
   from the exact frame that hallucinated a different room, the new model produces a soft,
   blurry, *same-scene* prediction — regression-style OOD degradation instead of
   mode-snapping. (Trained a small token→RGB decoder purely so humans can watch it think.)
- Figures: Gate A curves (pod: `results/dist_harness_nearchair/`), weld overlay,
  C0 table, hamper hallucination before/after strips (pod: `results/hamper_retest_*.png`).

**Closing beat:** 2026-06-11, on-robot: **3/3 physical arrivals — including ×2 on the exact
goal the old stack failed** — monotone descent 0.32 → 0.19, committed full-speed driving
far out, millimeter corrections near. The Gate A prediction held on hardware.

## Act IV — The graph: navigating beyond what the planner can see

**Beat:** the metric gives ~40 cm of vision. The room is 3 m. Route through memories.

Content:
1. **The idea in one line:** every frame the robot has ever seen is a place it has
   provably been; string them into a directed graph and let Dijkstra do the long-range
   thinking, so CEM only ever chases a waypoint one reach away (the regime that just went
   3/3).
2. **Build (all offline, zero new data):** cache DINOv2 tokens for all 4,500 chunk
   boundaries; **calibrate** rather than guess — within-episode pairs k chunks apart give
   the distance-per-chunk curve (k=1 → 0.092, k=3 → 0.182, far plateau 0.454); τ = 0.182
   = "one CEM reach." Temporal edges (the robot drove them — certified) + shortcut welds
   where different episodes pass through the same view (d < τ, cross-episode). 50
   disconnected threads → one connected component covering the room.
   - Reuse the existing explainer animation here (`context/figures/subgoal-graph-anim.mp4`
     and the 3-panel static) — this is exactly what it was made for.
3. **The two humbling catches (give these their own subsection — robots are spicy):**
   - **The graph must be directed.** First routes happily sent waypoints *backwards along
     episode threads*. The robot cannot drive backwards (no reverse in the data, vx ≥ 0
     clamp). An undirected edge encodes a lie about capability. Temporal edges became
     one-way.
   - **Even the welds lie about direction.** A weld at τ can hide ~10 cm of pose offset —
     sometimes *behind* you. Tightening the threshold collapsed the graph (τ=0.10 → 32%
     connectivity). The fix is the cleverest trick in the project: **motion-parallax
     direction certification** — for weld i→j, check whether i's *temporal successors* get
     closer to j; if yes, j is provably ahead. Zero new data; uses the trajectories as
     their own ground truth. 17,796 directed welds, 94.5% strongly-connected, direction
     guaranteed where the data can prove it.
4. **Runtime:** localize live frame by k-NN over the cache (**11 ms** — free next to the
   7 s plan), walk the goal-rooted shortest-path tree, hand CEM the waypoint's *real
   cached frame* as its goal; ENDGAME switches to the actual goal photo. Waypoint
   lookahead measured, not tuned by vibes: one-step descent reliability vs chunk gap
   (95.8% @ 2 chunks → 75.8% @ 10) → space waypoints at the 90% point.
5. **On-robot, with the fix-list told honestly:**
   - First attempt: localization flip-flopped between lookalike episode threads, routes
     re-rolled every replan, robot dithered. Fix: hysteresis — track along the committed
     path, demand a margin before rerouting.
   - Second: waypoints too close → timid near-zero commands. Fix (operator's call):
     waypoint floor — give CEM a *visibly different* target and it commits.
   - **Then: REACHED.** nearpurifier across the room — 129 steps, 40-hop route, tracked
     localization throughout, endgame at step 116, final dist 0.08 ≤ threshold. First
     end-to-end success of the full pipeline.
6. **The A/B that frames the whole act:** without the graph, the flat planner arrives
   from start-dist 0.35 but plateau-wanders forever from 0.45. The measured basin edge
   (0.35–0.45) matches the offline calibration. The graph is precisely the thing that
   crosses it.
- Figures: graph build animation (have), route filmstrips (pod:
  `results/subgoal_graph/route_*.png`), wormhole-audit montage (pod), on-robot
  rerun captures from `mpc_semantic_graph_nearpurifier4.rrd`, A/B traces
  (graph_dist monotone vs dist flat).

## Act V — What we built, what it cost, what's next (close)

Content:
1. **The system, restated as three layers** (one diagram):
   graph (topological memory, routes the room) → CEM + world model (local planner,
   ~40 cm of vision) → [future: visual servo (final centimeters)].
   Each layer keeps the next inside its comfort zone.
2. **The scoreboard:** 50 episodes / ~25 min of data; 160M WM trained overnight on one
   H100; frozen DINOv2 (zero metric training); graph built offline in minutes; 7 s
   replans; cross-room goal reached that the flat planner provably cannot.
3. **Honest limitations:** single room, single camera, stop-and-plan (not real-time),
   cross-session goal-image offset (~0.2 floor — recapture or threshold), forward-only,
   endgame convergence is goal-image-dependent (neardesk hovered).
4. **The lessons list (the share-bait section, each one earned in-text):**
   - The objective is part of the planner. Search was never broken; the metric was blind.
   - Make the bottleneck a number before changing the architecture (Gate A).
   - Judge world models by rollouts, not val loss.
   - A pooled correlation can hide a detectable signal; controlled contrasts win.
   - OOD failure *mode* matters as much as OOD failure: regression blurs, diffusion
     confidently teleports.
   - Topology is cheaper than capability: a graph fixed what no planning knob could.
   - Your graph encodes your robot's physics: no reverse ⇒ directed edges.
   - Real-robot debugging is mostly measurement design (tape measure > theory).
5. **Next chapter (published as future work, not blockers):** full-room recollection +
   retrain (C2: multi-camera, reverse segments — which literally adds edges to the graph),
   visual-servo endgame for the final centimeters (can use strafe + reverse since it
   bypasses the WM), inference speedup 7 s → ~1 s.
6. Repo link, acknowledgments (NanoWM, LeRobot/LeKiwi, DINO-WM/RAE-NWM lineage).

---

# Asset inventory

## Have (in repo)
- `context/figures/room.jpg` — the arena
- `context/figures/subgoal-graph-viz.png` — 3-panel graph explainer
- `context/figures/subgoal-graph-anim.mp4` (+ gif) — animated build/route/drive
- `context/figures/live-distribution-gap_*.png` — the hallucination montage
- `goals/*/goal.png` — actual goal images (show 2–3 inline)

## Have (on pod volume — pull before it dies)
- `results/dist_harness_nearchair/` — Gate A curves + gate_report (THE table's figures)
- `results/dist_harness_dino/`, `results/sweep_nearchair_imagined_dino/` — weld before/after
- `results/hamper_retest_*.png`, `results/c1_smoke_strip.png` — hallucination fixed strips
- `results/c0_diag_C0*/action_diagnostic.png` — C0 probe panels
- `results/subgoal_graph/route_*.png` + `audit/*.png` — route filmstrips, wormhole audit
- `results/offline_planning_*/montages/` — CEM-lands-on-goal montages
- `.rrd` files — source for screen-recordings: `mpc_semantic_graph_nearpurifier4.rrd`
  (the success), `mpc_semantic_nograph_nearhamper2.rrd` (the failure baseline),
  `mpc_nearfan2_thresh25.rrd` (first convergence)
- `viz/stationary-vs-translation/` — Act I refutation figures

## Need to make
- System diagram (3-layer architecture; one clean SVG)
- Timeline/arc graphic (optional but strong: the 12-day diagnosis→pivot→success arc)
- **The hero video:** side-by-side A/B — flat planner wandering vs graph run reaching
  (from .rrd replays or, better, a phone video of the robot + viewer picture-in-picture)
- Phone/room video of the robot actually driving (the only thing no .rrd can replace —
  needs one more robot session, aligns with the planned demo set: hamper A/B, purifier #2,
  neardesk)
- Gate A table restyled; C0 table restyled

## Demo set still to record on-robot (already planned in the log)
- nearhamper from far: baseline fails on tape + graph succeeds — **the strongest single asset**
- purifier from a new corner (#2), neardesk with the sparse-waypoint config

---

# Structure & platform notes (to decide later, recorded now)

- Length estimate: 4,500–6,500 words + ~15 figures + 2–3 videos. Long but skimmable:
  every act has a bolded TL;DR beat and a closing question; tables carry the evidence.
- Standalone site pros: video embeds, side-by-side layouts, the mp4 animation, custom
  hero. A simple static site (Astro/plain HTML on GitHub Pages or Vercel) is enough; no
  framework needed for one long page + assets.
- Cross-post strategy: full piece on the site; LinkedIn post = the hero video + 5 lessons
  + link; optional HN "Show HN".
- Possible cuts if too long: Act I items 1–2 can compress to one paragraph + sidebar;
  bring-up grit → sidebar box; literature interlude → footnotes.
- Keep ops war stories as styled sidebars ("Field notes"): deg/s units, omniwheel
  wheels-up illusion, USB camera identity swap, the 30-second host window, the rsync
  near-disaster saved by an NFS snapshot. They humanize without derailing.
