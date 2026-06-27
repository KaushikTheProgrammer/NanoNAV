"""
Diffusion Forcing concept diagram — modern animated SVG.

Top panel    — Standard diffusion: every frame shares one noise level, denoised
               in lockstep (grain pulses in unison, no frame is ever clean alone).
Bottom panel — Diffusion Forcing: each frame carries its own noise level, so the
               context frames stay clean (sigma=0) while the target frame visibly
               denoises and a next frame is predicted — the autoregressive loop.

Self-contained animated SVG. SMIL <animate> (grain opacity + noise-bar height)
and CSS keyframes run inside an <img>, so the denoising is visible on the
deployed page with no JS and stays crisp at any zoom.
"""
import os

BLUE_EC, BLUE_TX = "#3b7dc4", "#163a60"
AMBR_EC, AMBR_TX = "#d8902a", "#7a4d0c"
PURP_EC, PURP_TX = "#7c4dbc", "#4a2c7a"
NOISE = "#e0654f"
INK = "#1a1f2b"
MUTE = "#6b7280"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

W, H = 1160, 632
FW, FH, GAP, X0 = 138, 96, 16, 49
STEP = FW + GAP

# scene gradients: (sky, floor, object, border, text)
KIND = {
    "std": ("#f1f2f5", "#e1e3ea", "#b6bcc8", "#9aa1ad", INK),
    "ctx": ("#eef5fd", "#dae8fb", "#6aa3df", BLUE_EC, BLUE_TX),
    "tgt": ("#fff6e8", "#fde6c4", "#e0922f", AMBR_EC, AMBR_TX),
    "pred": ("#f4eefc", "#e6daf9", "#9a6fd0", PURP_EC, PURP_TX),
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
                 f'fill="{NOISE if kind=="std" else "#c9ced8"}" '
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
    '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f6f7fb"/>'
    '</linearGradient>')
parts.append(f'<clipPath id="fclip"><rect width="{FW}" height="{FH}" rx="12"/></clipPath>')
parts.append(
    '<filter id="grain"><feTurbulence type="fractalNoise" baseFrequency="0.9" '
    'numOctaves="3" seed="7" stitchTiles="stitch"/>'
    '<feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  '
    '0.42 0.42 0.42 0 0"/>'
    '</filter>')
parts.append(
    f'<marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" '
    f'orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L7.5,3 L0,6 Z" fill="{PURP_EC}"/></marker>')
parts.append('</defs>')

# ── card + title ───────────────────────────────────────────────────────────
parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e6e8ee" stroke-width="1.5"/>')
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
parts.append(f'<line x1="40" y1="324" x2="{W-40}" y2="324" stroke="#e6e8ee" stroke-width="1.3"/>')

# ════════════════════════════ PANEL B — forcing ═══════════════════════════
by = 372
parts.append(f'<text x="{X0}" y="350" font-size="13.5" font-weight="700" fill="{INK}">'
             f'Diffusion Forcing <tspan font-weight="400" fill="{MUTE}" font-style="italic">'
             f'— independent noise per frame → autoregressive rollout</tspan></text>')

# 5 clean context frames
for i in range(5):
    fx = X0 + i * STEP
    parts.append(frame(fx, by, "ctx", f"f{i+1}", BLUE_EC))
    parts.append(f'<text x="{fx+FW/2}" y="{by-12}" text-anchor="middle" font-size="11.5" '
                 f'font-style="italic" fill="{BLUE_EC}">context</text>')

# target frame (denoising)
tgt_x = X0 + 5 * STEP
tgt_anim = ('<animate attributeName="opacity" values="0.78;0.78;0;0;0.78" '
            'keyTimes="0;0.18;0.55;0.86;1" dur="4s" repeatCount="indefinite"/>')
parts.append(frame(tgt_x, by, "tgt", "f6", AMBR_EC, noise=(0.78, tgt_anim)))
parts.append(f'<text x="{tgt_x+FW/2}" y="{by-12}" text-anchor="middle" font-size="11.5" '
             f'font-style="italic" fill="{AMBR_EC}">target · denoising</text>')

# predicted frame (fades in once target is clean)
pred_x = tgt_x + STEP + 16
parts.append('<g opacity="0"><animate attributeName="opacity" values="0;0;1;1;0" '
             'keyTimes="0;0.55;0.66;0.9;1" dur="4s" repeatCount="indefinite"/>'
             + frame(pred_x, by, "pred", "f&#770;7", PURP_EC, dashed=True)
             + f'<text x="{pred_x+FW/2}" y="{by-12}" text-anchor="middle" font-size="11.5" '
               f'font-style="italic" fill="{PURP_EC}">predicted</text>'
             + '</g>')

# arrow target -> predicted
parts.append('<g opacity="0"><animate attributeName="opacity" values="0;0;1;1;0" '
             'keyTimes="0;0.5;0.66;0.9;1" dur="4s" repeatCount="indefinite"/>'
             f'<path d="M{tgt_x+FW+4},{by+FH/2} L{pred_x-4},{by+FH/2}" fill="none" '
             f'stroke="{PURP_EC}" stroke-width="1.8" marker-end="url(#ah)"/>'
             f'<text x="{(tgt_x+FW+pred_x)/2}" y="{by+FH/2-8}" text-anchor="middle" '
             f'font-size="10.5" font-style="italic" fill="{PURP_EC}">predict next</text>'
             '</g>')

# noise bars row B
bbase = by + FH + 48
for i in range(5):
    cx = X0 + i * STEP + FW / 2
    parts.append(bar(cx, bbase, "ctx", "σ = 0", BLUE_EC))
# target bar shrinking
tb = ('<animate attributeName="height" values="38;38;0;0;38" keyTimes="0;0.18;0.55;0.86;1" '
      'dur="4s" repeatCount="indefinite"/>'
      '<animate attributeName="y" values="%d;%d;%d;%d;%d" keyTimes="0;0.18;0.55;0.86;1" '
      'dur="4s" repeatCount="indefinite"/>' % (bbase-38, bbase-38, bbase, bbase, bbase-38))
parts.append(bar(tgt_x + FW / 2, bbase, "tgt", "σ: 1→0", AMBR_EC, anim=tb))
parts.append(bar(pred_x + FW / 2, bbase, "pred", "σ→0", PURP_EC))
parts.append(f'<text x="{X0-10}" y="{bbase-8}" text-anchor="end" font-size="10.5" '
             f'font-style="italic" fill="{NOISE}">noise</text>')

# ── legend ─────────────────────────────────────────────────────────────────
leg = [("#dae8fb", BLUE_EC, "Context frame (σ = 0, clean)", 250),
       ("#fde6c4", AMBR_EC, "Target frame (denoising)", 560),
       ("#e6daf9", PURP_EC, "Predicted next frame", 830)]
for fc, ec, txt, x in leg:
    parts.append(f'<rect x="{x}" y="{H-26}" width="18" height="13" rx="3" '
                 f'fill="{fc}" stroke="{ec}" stroke-width="1.5"/>')
    parts.append(f'<text x="{x+25}" y="{H-16}" font-size="12.5" fill="{INK}">{txt}</text>')

parts.append('</svg>')

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'diffusion_forcing.svg')
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
