"""
Diffusion Forcing concept diagram — modern animated SVG.

Top panel    — Standard diffusion: every frame shares ONE noise level. The grain
               and the noise bars move in lockstep (always identical sigma).
Bottom panel — Diffusion Forcing: every frame carries its OWN independent noise
               level. Each frame's grain + sigma bar animate on a different clock
               (different amplitude, period and phase), so the bars bob out of
               sync — at any instant the six sigmas are all different. That
               independence (some frames clean, some noisy, freely mixed) is what
               lets the model keep clean frames as context and denoise the next,
               i.e. roll out autoregressively at inference.

Self-contained animated SVG. SMIL <animate> (grain opacity + noise-bar height)
runs inside an <img>, so the motion is visible on the deployed page with no JS
and stays crisp at any zoom.
"""
import os

# warm cream/terracotta palette to match the site (style.css):
# bg #fdf6e3 · accent #b3552c · tan #f5ead0. Muted teal = context/clean,
# terracotta accent = target/denoising, dusty plum = predicted.
BLUE_EC, BLUE_TX = "#5d8a8a", "#2d4a4a"
AMBR_EC, AMBR_TX = "#b3552c", "#7a3a1a"
PURP_EC, PURP_TX = "#845a8e", "#523460"
NOISE = "#c0613c"
INK = "#2c2722"
MUTE = "#7c7565"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

W, H = 1160, 632
FW, FH, GAP, X0 = 138, 96, 16, 49
STEP = FW + GAP

# scene gradients: (sky, floor, object, border, text)
KIND = {
    "std":  ("#eee9de", "#e3ddcc", "#b3aa95", "#a99f86", INK),
    "ctx":  ("#e3eeee", "#d3e3e3", "#6f9c9c", BLUE_EC, BLUE_TX),
    "tgt":  ("#fbeede", "#f4dcc0", "#cf7f44", AMBR_EC, AMBR_TX),
    "pred": ("#efe6f0", "#e2d2e6", "#a07caa", PURP_EC, PURP_TX),
}


def frame(fx, fy, kind, label, lblcolor, noise=None, dashed=False):
    sky, floor, obj, ec, tx = KIND[kind]
    gid = f"g{kind}{fx}"
    s = [f'<g transform="translate({fx},{fy})">']
    s.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{sky}"/>'
             f'<stop offset="1" stop-color="{floor}"/></linearGradient>')
    s.append(f'<g clip-path="url(#fclip)">')
    s.append(f'<rect width="{FW}" height="{FH}" fill="url(#{gid})"/>')
    s.append(f'<line x1="0" y1="{FH*0.62:.0f}" x2="{FW}" y2="{FH*0.62:.0f}" '
             f'stroke="{obj}" stroke-width="1.4" opacity="0.55"/>')
    s.append(f'<circle cx="{FW*0.66:.0f}" cy="{FH*0.40:.0f}" r="11" fill="{obj}" opacity="0.9"/>')
    if noise is not None:
        op, anim = noise
        s.append(f'<rect width="{FW}" height="{FH}" filter="url(#grain)" '
                 f'opacity="{op}">{anim}</rect>')
    s.append('</g>')
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    s.append(f'<rect width="{FW}" height="{FH}" rx="12" fill="none" '
             f'stroke="{ec}" stroke-width="2"{dash}/>')
    s.append(f'<text x="{FW/2}" y="{FH*0.40+6:.0f}" text-anchor="middle" '
             f'font-size="18" font-style="italic" font-weight="600" fill="{tx}" '
             f'opacity="0.92">{label}</text>')
    s.append('</g>')
    return "\n".join(s)


def bar(cx, base, kind, sigma_lbl, sigma_color, anim=None):
    s = []
    if anim is None:
        h = 22 if kind == "std" else 2
        s.append(f'<rect x="{cx-22}" y="{base-h}" width="44" height="{h}" rx="2" '
                 f'fill="{NOISE if kind=="std" else "#cdc4b0"}" '
                 f'opacity="{0.8 if kind=="std" else 1}"/>')
    else:
        s.append(f'<rect x="{cx-22}" width="44" rx="2" fill="{NOISE}" opacity="0.82">'
                 f'{anim}</rect>')
    s.append(f'<text x="{cx}" y="{base+18}" text-anchor="middle" font-size="11.5" '
             f'fill="{sigma_color}">{sigma_lbl}</text>')
    return "\n".join(s)


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}" role="img" aria-label="Diffusion forcing vs standard diffusion">'
]

# ── defs ───────────────────────────────────────────────────────────────────
parts.append('<defs>')
parts.append(
    '<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#fdf7e9"/><stop offset="1" stop-color="#f6ecd2"/>'
    '</linearGradient>')
parts.append(f'<clipPath id="fclip"><rect width="{FW}" height="{FH}" rx="12"/></clipPath>')
parts.append(
    '<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.9" '
    'numOctaves="3" seed="7" stitchTiles="stitch"/>'
    '<feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  '
    '0.42 0.42 0.42 0 0"/>'
    '</filter>')
parts.append(
    '<linearGradient id="nscale" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#efe2d2"/><stop offset="1" stop-color="#a8451f"/>'
    '</linearGradient>')
parts.append('</defs>')

# ── card + title ───────────────────────────────────────────────────────────
parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e7dcc0" stroke-width="1.5"/>')
parts.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="19" '
             f'font-weight="700" fill="{INK}">Diffusion Forcing vs Standard Diffusion</text>')

# ════════════════════════════ PANEL A — standard ══════════════════════════
ay = 96
parts.append(f'<text x="{X0}" y="74" font-size="13.5" font-weight="700" fill="{INK}">'
             f'Standard Diffusion <tspan font-weight="400" fill="{MUTE}" '
             f'font-style="italic">— one shared noise level, denoised in lockstep</tspan></text>')
unison = ('<animate attributeName="opacity" values="0.6;0.6;0.28;0.6;0.6" '
          'keyTimes="0;0.2;0.5;0.8;1" dur="4s" repeatCount="indefinite"/>')
for i in range(6):
    fx = X0 + i * STEP
    parts.append(frame(fx, ay, "std", f"f{i+1}", MUTE, noise=(0.6, unison)))
abase = ay + FH + 48
ub = ('<animate attributeName="height" values="22;22;10;22;22" keyTimes="0;0.2;0.5;0.8;1" '
      'dur="4s" repeatCount="indefinite"/>'
      '<animate attributeName="y" values="%d;%d;%d;%d;%d" keyTimes="0;0.2;0.5;0.8;1" '
      'dur="4s" repeatCount="indefinite"/>' % (abase-22, abase-22, abase-10, abase-22, abase-22))
for i in range(6):
    cx = X0 + i * STEP + FW / 2
    parts.append(bar(cx, abase, "std", "σ = 0.6", NOISE, anim=ub))
parts.append(f'<text x="{X0-10}" y="{abase-8}" text-anchor="end" font-size="10.5" '
             f'font-style="italic" fill="{NOISE}">noise</text>')

# divider
parts.append(f'<line x1="40" y1="324" x2="{W-40}" y2="324" stroke="#e7dcc0" stroke-width="1.3"/>')

# ════════════════════════════ PANEL B — forcing ═══════════════════════════
# Each frame gets its OWN noise level, animated on its own clock so the six
# sigmas drift out of sync: (min sigma, max sigma, period s, phase offset s).
by = 372
bbase = by + FH + 48
MAXBAR = 42
parts.append(f'<text x="{X0}" y="350" font-size="13.5" font-weight="700" fill="{INK}">'
             f'Diffusion Forcing <tspan font-weight="400" fill="{MUTE}" font-style="italic">'
             f'— an independent noise level per frame</tspan></text>')

indep = [
    (0.08, 0.55, 3.0, 0.0),
    (0.62, 1.00, 4.2, 1.1),
    (0.00, 0.28, 2.4, 0.6),
    (0.34, 0.82, 3.6, 2.0),
    (0.18, 0.92, 5.0, 0.4),
    (0.46, 0.96, 2.8, 1.7),
]
SPL = 'calcMode="spline" keyTimes="0;0.5;1" keySplines="0.4 0 0.6 1;0.4 0 0.6 1"'
for i, (a, b, period, off) in enumerate(indep):
    fx = X0 + i * STEP
    cx = fx + FW / 2
    # grain overlay opacity tracks sigma, on its own clock
    op_anim = (f'<animate attributeName="opacity" values="{a*0.82:.3f};{b*0.82:.3f};{a*0.82:.3f}" '
               f'dur="{period}s" begin="-{off}s" repeatCount="indefinite" {SPL}/>')
    parts.append(frame(fx, by, "std", f"f{i+1}", MUTE, noise=(a * 0.82, op_anim)))
    # sigma bar, same independent clock
    ha, hb = a * MAXBAR, b * MAXBAR
    bar_anim = (
        f'<animate attributeName="height" values="{ha:.1f};{hb:.1f};{ha:.1f}" '
        f'dur="{period}s" begin="-{off}s" repeatCount="indefinite" {SPL}/>'
        f'<animate attributeName="y" values="{bbase-ha:.1f};{bbase-hb:.1f};{bbase-ha:.1f}" '
        f'dur="{period}s" begin="-{off}s" repeatCount="indefinite" {SPL}/>')
    lbl = f'σ<tspan baseline-shift="sub" font-size="8">{i+1}</tspan>'
    parts.append(bar(cx, bbase, "tgt", lbl, NOISE, anim=bar_anim))
parts.append(f'<text x="{X0-10}" y="{bbase-8}" text-anchor="end" font-size="10.5" '
             f'font-style="italic" fill="{NOISE}">noise</text>')
parts.append(f'<text x="{X0 + 6*STEP - GAP}" y="350" text-anchor="end" font-size="11" '
             f'font-style="italic" fill="{MUTE}">clean frames can be context while the next '
             f'is denoised</text>')

# ── legend — noise-level scale ─────────────────────────────────────────────
gx, gw, gy = 505, 150, H - 26
parts.append(f'<text x="{gx-12}" y="{gy+11}" text-anchor="end" font-size="12" '
             f'fill="{INK}">σ = 0 clean</text>')
parts.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="14" rx="3" '
             f'fill="url(#nscale)" stroke="#d8c7ab" stroke-width="1"/>')
parts.append(f'<text x="{gx+gw+12}" y="{gy+11}" font-size="12" '
             f'fill="{INK}">σ = 1 noisy</text>')

parts.append('</svg>')

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'diffusion_forcing.svg')
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
