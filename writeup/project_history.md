# NanoNAV — Project History

The full project narrative, retold as a continuous story from the experiment-log
chronology (2026-06-01 → 2026-06-13). This is the validated spine for the write-up.
Source of truth for every number: `context/experiment-log.md`.

---

## Background — where this came from
A world model is, roughly, a learned simulator: feed it the recent past and a candidate action,
and it predicts what comes next — usually as future observations rather than abstract state. But
"world model" is doing a lot of work in one phrase, and Fei-Fei Li's [functional taxonomy of world
models](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models) is the cleanest cut
I've seen. She splits them by *what they output*: a **Renderer** outputs pixels meant for human
eyes, where visual fidelity is the thing that matters; a **Simulator** outputs state — a
geometrically and physically faithful representation that programs (and learning agents) can
compute on; and a **Planner** outputs *actions* — given observations and a goal, it decides what
the agent should do next, closing the perception–action loop. She argues these eventually converge
into unified models, with simulation as the linchpin.

This project lives squarely in the **Planner** corner of that taxonomy. It is not trying to render
a beautiful world or to be a faithful physics engine — its imagined frames are, frankly, blurry. It
is trying to use a small, imperfect imagination as the inner loop of a controller: propose an
action, imagine its consequence, score that consequence against a goal image, act. The whole story
that follows is what happens when you take the Planner ambition seriously on cheap hardware and tiny
data — and discover that the hard part isn't the imagining, it's making the *score* mean something.

The spark was the **Nano World Models** project and, more than the code, its framing: a call to
*democratize* world-model research — to show the ideas don't require a frontier lab's compute or
data to be worth building on. That resonated, and it set the constraint that defines everything
here: do this small.

What kept pulling me toward robotics specifically is the part of the field I love most — taking an
idea, an algorithm, a piece of software, and making something *happen in the physical world*. A lot
of the world-model literature I was reading stopped short of that. The work was excellent, but it
often lived entirely in simulation — **DINO-WM**, for instance, runs all of its experiments in sim.
I wanted to see whether the planning-world-model recipe could survive contact with a real robot, a
real room, and real cameras, where the failures are messier and the wins are undeniable.

The other observation that nudged me forward was noticing that the Nano World Models work trained
and evaluated on the **RT-1 dataset** — a robotics dataset, not a game or a driving sim. If the
recipe worked on robot data at that scale, maybe it would work on *my* robot data.

And I happened to have the robot. A **LeKiwi** was sitting on my desk from earlier imitation-
learning projects — a small mobile manipulator from the LeRobot ecosystem, the kind of hardware
that makes "do this small" actually possible. The idea more or less wrote itself: collect my own
data, train a nano-scale world model on it, and see if I could plan with it on the real machine.
That's how the journey started.

---

## The premise

The goal was deliberately stark: drive a robot to a place you specify with a **photograph** —
no map, no SLAM, no localization, no GPS, no reward function, no task demonstrations. Just a
current camera frame and a goal image, in, and wheel velocities out. The bet behind it was that
a world model trained on raw experience could absorb everything the classical stack does for this
one task, and the *interesting* version of that bet was making it work on a comically small
budget: **50 teleop episodes, ~25 minutes of driving, 44,926 frames, one room, one overhead
camera, one consumer GPU.** Small data wasn't a compromise to apologize for; it was the whole
point.

The design decisions settled fast and held all the way through: action is a body-frame
**(Δx, Δθ)** chunk — heading-invariant, only 2-D so CEM searches 6 dimensions at horizon 3
instead of 30; Δy was proven droppable (max 0.58 mm/chunk, *measured*, smaller even than the
back-of-envelope said). The camera is an elevated ~55° third-person view with the robot's own
body in frame as an ego-motion reference. The data is *exploratory driving, not demonstrations* —
tasks enter only at inference, and bad trajectories are valuable because CEM has to score and
reject bad actions. The very first validation already drew blood: `theta.vel` is **degrees per
second**, and integrating it as radians spins one episode through 7,528° of phantom rotation.
Units bugs are navigation bugs — a theme that recurs.

## Act I — A world model that ignores its own actions

The first trained model (Run 001) failed the most basic test: feed it zero actions, the true
actions, or random actions, and it predicts nearly the same future regardless (action-embed RMS
0.0088; zero ≈ random). A world model that ignores actions is a screensaver.

**This is where the project's signature pattern appears for the first time: a confident wrong
diagnosis, killed by a controlled measurement.** The frame-interval sweep showed correlation
between translation magnitude and latent change was ≈0 at every setting, and the conclusion
nearly written into the record was *"this camera geometrically cannot see forward motion."* The
refutation came the next day from a *controlled contrast* — hold rotation near zero, compare
stationary vs pure-translation chunks — and translation separated cleanly, AUC 0.94–0.98, with a
textbook dose-response as the frame interval grew. The signal was in the latent the whole time;
the pooled correlation was just the wrong estimator (bang-bang speeds, rotation chunks polluting
it). The real fix was training-side: Run 002 at f=10 produced a live, action-sensitive branch
(clean gt < zero < random), and a second lesson fell out — for a diffusion-forcing model,
**val_loss lies**: rollout quality is U-shaped in training step, the val-best checkpoint (4125)
is *not* the best roller, ~8K peaks, 12K overfits. Judge by rollouts, carry **step-8000**.

Offline, the planner then passed everything: CEM hits the world model's own ceiling
(reached_ratio ~1.0 in every motion bucket), recovers the true commands (sign 100%, ~1 cm / ~2.5°
error), and the cheap DDIM=3 sampler holds at ~7 s/replan. **Remember this — the search engine
was validated early and was never the thing that broke.**

## Act II — The robot says no (three wrong root causes)

On hardware, offline-perfect became on-robot-wandering. The bring-up itself was honest grit: the
units/sign contract pinned on the real robot, the discovery that wheels-up testing literally
cannot show body rotation (omniwheels spin tangentially), open-loop replay matching dead-reckoning
to ~0 cm even through a 117° arc. Then the first closed-loop run: planning executed correctly, but
distance-to-goal hovered ~45 for 22 steps while the robot wiggled in place, and the Pi host
dropped mid-run.

The debugging here is the heart of the story because **two more confident diagnoses both turned
out to be over-reach:**

- **A real bug, but not the cause:** the live preprocessor fed the VAE pixels in [0,1] while
  training used [−1,1]. Fixed — still no convergence. Necessary, not sufficient.
- **Wrong diagnosis #2:** a drive-straight probe showed distance flat for 46 cm then snapping down
  after a heading nudge, and this got written up as a whole theory — *"the wide-angle camera
  produces a flat, poorly-conditioned objective; camera and objective are a joint design
  failure."* Retracted **the same day** by a controlled hand-placed radial sweep: latent distance
  is actually *monotone* over 40 cm with SNR ~17σ/10 cm. The "flat 46 cm" was the robot drifting
  *off-axis* — path length, not radial approach.

Re-capturing the goal at the true pose produced the **first convergence — REACHED ×2** in 10–14
steps. The whole stack vindicated… for goals already near where the robot drove. But far goals
still stalled, and *no knob helped* — horizon 5, wider sampling variance, higher speed cap, all
nothing. The decisive qualitative observation, the operator watching the robot: CEM **correctly
turned the bot to face the chair**, and the distance metric **barely rewarded it.** The search was
choosing the right behavior; the objective couldn't see it. Then it got worse — from under-covered
poses the world model **hallucinated**, its imagined rollout snapping to a confident, vivid view
of *a different part of the room*. Off-distribution, a diffusion model doesn't degrade gracefully;
it teleports. With a hallucinated latent, both the distance readout and what CEM optimizes are
garbage.

The bottom of the arc: two coupled blockers — the objective is blind far from the goal, and the
world model hallucinates off-coverage — and every cheap fix exhausted.

## Act III — Turn the bottleneck into a number, then pivot

This is the turn. Instead of arguing about the objective, **measure it.** Gate A: hand-place the
robot at tape-measured displacements (radial 10–60 cm, lateral ±60 cm, yaw ±30°), and grade every
candidate distance on identical frames — Spearman ρ, far-band slope against the robot's own
standing-still noise, yaw basin. The result is the cleanest punchline in the project: the SD-VAE
latent-L2 objective we'd been planning with was perfectly *ordered* (ρ=1.0) but its 40–60 cm
gradient sat **below the standing-still noise floor** (1.25σ vs a 3σ gate). CEM literally could
not see far-field progress. Meanwhile **frozen, never-trained DINOv2 patch cosine** had 12–21σ of
gradient on the *same pixels*. The information was always there; the representation buried it.

Then the **pivot (Option C):** don't bolt the better metric onto the old stack (a weld test showed
the old WM's imagined latents sit +23σ off the real curve — its imagination is unreliable too).
Instead **retrain the world model to predict frozen DINOv2 patch tokens**, so the rollout space
*is* the validated distance space and the planning cost (token cosine) needs zero training. The
literature backed the shape (ViNG/GNM/ViNT: distance head + graph is the field recipe; DINO-WM and
RAE-NWM plan on exactly this token cost, RAE-NWM beating the VAE-latent equivalent 79% vs 43% on
Habitat), while our sweep measured the thing no paper had — 0–60 cm flatness at robot scale.

And the most satisfying sub-result: the C0 probe matrix dissolved a community "finding." The
received wisdom (NanoWM's own Table 5/6) was that semantic latents kill action conditioning. Four
3k-step runs, one variable at a time, showed it was never the latents — it was the **injection
path**: additive conditioning atrophies to RMS 0.0028 (reproducing the documented failure *on
demand*), while AdaLN-fuse on the *same data and latents* holds 0.207; x0 + AdaLN-fuse wins. A
field belief, reproduced, narrowed, and explained in twelve GPU-hours. Gate C then showed the new
model's action branch *strengthening* with training (RMS 0.333, margin 43.4), CEM back at the WM
ceiling, the weld ρ recovering 0.29 → 0.876, and — critically — **the hallucination fixed at the
source**: from the exact frame that once conjured a different room, the new model produces a soft,
blurry, *same-scene* prediction. Regression blurs where diffusion teleported. (A small token→RGB
decoder was trained purely so humans can watch it think.)

The payoff landed on hardware on June 11: **3/3 physical arrivals, including ×2 on the exact goal
the old stack had failed**, monotone descent 0.32 → 0.19, committed full-speed driving far out,
millimeter corrections near. The Gate A prediction held in the real room.

## Act IV — A graph of remembered moments

The new metric buys ~40 cm of usable vision; the room is 3 m. The idea that closes the gap:
**every frame the robot ever saw is a place it provably reached** — cache the DINOv2 tokens for
all 4,500 chunk boundaries, string them into a graph, and let Dijkstra do the long-range reasoning
so CEM only ever chases a waypoint *one reach away* (the regime that just went 3/3). Nothing is
guessed: the edge threshold τ=0.182 is *calibrated* from within-episode pairs (k chunks apart →
measured distance), and waypoint spacing comes from a measured one-step-descent reliability curve
(95.8% at 2 chunks, falling to 75.8% at 10) → space at the 90% point. Fifty disconnected episode
threads weld into one connected component.

Then two humbling catches, each worth its own beat because each is the robot's physics asserting
itself:

- **The graph must be directed.** The first routes cheerfully sent waypoints *backwards along
  episode threads* — but the robot **has no reverse** (forward-only data, vx ≥ 0 clamp). An
  undirected edge encodes a lie about what the robot can do. Temporal edges became one-way.
- **Even the welds lie about direction** — a weld at τ can hide ~10 cm of pose offset, sometimes
  *behind* you, and tightening the threshold collapsed connectivity (τ=0.10 → 32%). The fix is the
  project's cleverest trick: **motion-parallax direction certification** — for a candidate weld
  i→j, check whether i's own *temporal successors* get closer to j; if they do, j is provably
  ahead. Zero new data; the trajectories certify themselves. Result: 17,796 directed welds, 94.5%
  strongly connected, direction guaranteed where the data can prove it.

Runtime: localize the live frame by k-NN over the cache (**11 ms — free** next to a 7 s plan),
walk the goal-rooted shortest-path tree, hand CEM the waypoint's *real cached frame*, and switch
to the actual goal photo only in the endgame. The on-robot road there was, again, told honestly:
localization flip-flopped between lookalike episode threads (fix: hysteresis + committed-path
stickiness), then waypoints sat too close and CEM gave timid near-zero commands (fix, the
operator's call: a waypoint floor — give it a *visibly different* target and it commits).
**Then: REACHED nearpurifier across the room** — 129 steps, a 40-hop route, tracked the whole way,
endgame at step 116 closing 0.30 → 0.08. First end-to-end success of the full pipeline.

The A/B that frames the entire act: without the graph the flat planner arrives from start-distance
0.35 but plateau-wanders forever from 0.45; the measured basin edge (0.35–0.45) matches the
offline calibration exactly. The graph is precisely the thing that crosses it.

## The close — what 25 minutes of data bought

Three layers, each keeping the next inside its comfort zone: a **graph** (topological memory,
routes the room) feeding **CEM + world model** (local planner, ~40 cm of vision), with a
**visual-servo endgame** (the final centimeters) as the named future piece. The honest limits stay
in frame: one corner of one room, one camera, stop-and-plan not real-time, a cross-session
goal-image offset (~0.2 floor), forward-only motion, and a goal-image-dependent endgame floor
(nearpurifier closes to 0.08; neardesk hovered at 0.30). And the lessons, each earned in-text: the
objective is part of the planner; make the bottleneck a number before you change the architecture;
judge world models by rollouts; controlled contrasts beat pooled correlations; OOD failure *mode*
matters (blur vs teleport); topology is cheaper than capability; your graph encodes your robot's
physics; real-robot debugging is mostly measurement design.

---

**The spine, in one line:** *the search was never broken — the objective was blind, and the
project is the story of proving that with a tape measure and then fixing it twice, first by
changing what the model sees and then by routing through what it remembers.*
