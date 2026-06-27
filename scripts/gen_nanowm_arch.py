"""
NanoWM architecture diagram — modern animated SVG.

Frozen VAE context latents + action chunk -> Transformer (AdaLN) ->
predicted next latent -> token-to-RGB decoder (visualization only).

Output is a self-contained animated SVG. CSS @keyframes (flowing connectors,
glow) and SMIL <animateMotion> (data-flow pulses) both run even when the file
is loaded through an <img> tag, so it stays crisp at any zoom and animates on
the deployed page without any JS.
"""
import os

# ── palette ────────────────────────────────────────────────────────────────
BLUE_FC1, BLUE_FC2, BLUE_EC, BLUE_TX = "#eaf2fc", "#d3e4fa", "#3b7dc4", "#163a60"
AMBR_FC1, AMBR_FC2, AMBR_EC, AMBR_TX = "#fef1da", "#fde0b6", "#d8902a", "#7a4d0c"
GREN_FC1, GREN_FC2, GREN_EC, GREN_TX = "#e6f5e9", "#d0eed6", "#3a9d4e", "#1f5e2d"
WIRE = "#9aa3b2"
INK = "#1a1f2b"
MUTE = "#6b7280"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

W, H = 1160, 500


def box(x, y, w, h, fc1, fc2, ec, sw=1.7):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" ry="14" '
        f'fill="url(#{fc1})" stroke="{ec}" stroke-width="{sw}" '
        f'filter="url(#soft)"/>'
    )


def wire(d, cls="wire"):
    return f'<path class="{cls}" d="{d}" fill="none" marker-end="url(#ah)"/>'


def pulse(path_d, dur, begin="0s", color=AMBR_EC):
    # a moving dot that traces path_d (drawn invisibly, referenced by the dot)
    pid = f"p{abs(hash(path_d)) % 100000}"
    return (
        f'<path id="{pid}" d="{path_d}" fill="none" stroke="none"/>'
        f'<circle r="3.4" fill="{color}" class="dot">'
        f'<animateMotion dur="{dur}" begin="{begin}" repeatCount="indefinite" '
        f'rotate="auto" keyPoints="0;1" keyTimes="0;1" calcMode="linear">'
        f'<mpath href="#{pid}"/></animateMotion></circle>'
    )


parts = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}" role="img" aria-label="NanoWM architecture">'
)

# ── defs ───────────────────────────────────────────────────────────────────
parts.append('<defs>')
for name, c1, c2 in [("bl", BLUE_FC1, BLUE_FC2), ("am", AMBR_FC1, AMBR_FC2),
                     ("gr", GREN_FC1, GREN_FC2)]:
    parts.append(
        f'<linearGradient id="{name}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/>'
        f'</linearGradient>')
parts.append(
    '<linearGradient id="card" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f6f7fb"/>'
    '</linearGradient>')
parts.append(
    '<filter id="soft" x="-20%" y="-20%" width="140%" height="140%">'
    '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1a1f2b" flood-opacity="0.12"/>'
    '</filter>')
parts.append(
    '<filter id="blur"><feGaussianBlur stdDeviation="6"/></filter>')
parts.append(
    f'<marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" '
    f'orient="auto" markerUnits="userSpaceOnUse">'
    f'<path d="M0,0 L7.5,3 L0,6 Z" fill="{WIRE}"/></marker>')
parts.append(
    '<style>'
    '.wire{stroke:%s;stroke-width:1.8;stroke-linecap:round;'
    'stroke-dasharray:1 7;animation:flow 1.1s linear infinite;}'
    '@keyframes flow{to{stroke-dashoffset:-16;}}'
    '.glow{animation:glow 2.8s ease-in-out infinite;}'
    '@keyframes glow{0%%,100%%{opacity:.18}50%%{opacity:.6}}'
    '.dot{filter:drop-shadow(0 0 2px rgba(0,0,0,.15));}'
    '.lbl{fill:%s;font-size:13px;}'
    '.sub{fill:%s;font-size:11px;font-style:italic;}'
    '</style>' % (WIRE, INK, MUTE))
parts.append('</defs>')

# ── card backdrop + title ──────────────────────────────────────────────────
parts.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" '
             f'fill="url(#card)" stroke="#e6e8ee" stroke-width="1.5"/>')
parts.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="19" '
             f'font-weight="700" fill="{INK}">NanoWM Architecture</text>')

# ── camera glyph ───────────────────────────────────────────────────────────
parts.append(f'<text x="120" y="84" text-anchor="middle" class="sub">Camera frames</text>')
for i, off in enumerate((10, 5, 0)):
    parts.append(f'<rect x="{96-off}" y="{96+off}" width="48" height="30" rx="4" '
                 f'fill="#ffffff" stroke="{MUTE}" stroke-width="1.3" opacity="{0.5+0.2*i}"/>')

# ── frozen VAE encoder ─────────────────────────────────────────────────────
parts.append(box(40, 160, 160, 70, "bl", "bl", BLUE_EC))
parts.append(f'<text x="120" y="190" text-anchor="middle" font-size="13.5" '
             f'font-weight="700" fill="{BLUE_TX}">Frozen VAE Encoder</text>')
parts.append(f'<text x="120" y="210" text-anchor="middle" class="sub" '
             f'fill="{BLUE_TX}">SD-pretrained, never trained</text>')

# ── context latents strip ──────────────────────────────────────────────────
lat_y, lw = 312, 58
lxs = [40, 104, 168, 232]
subs = ["t−3", "t−2", "t−1", "t"]
for lx, s in zip(lxs, subs):
    parts.append(box(lx, lat_y, lw, 58, "bl", "bl", BLUE_EC))
    parts.append(
        f'<text x="{lx+lw/2}" y="{lat_y+35}" text-anchor="middle" font-size="16" '
        f'font-style="italic" fill="{BLUE_TX}">z'
        f'<tspan baseline-shift="sub" font-size="10">{s}</tspan></text>')
parts.append(f'<text x="175" y="300" text-anchor="middle" class="sub" '
             f'fill="{BLUE_TX}">context latents</text>')

# ── transformer ────────────────────────────────────────────────────────────
tx, ty, tw, th = 420, 158, 272, 214
parts.append(f'<rect x="{tx-8}" y="{ty-8}" width="{tw+16}" height="{th+16}" rx="20" '
             f'fill="none" stroke="{AMBR_EC}" stroke-width="6" class="glow" filter="url(#blur)"/>')
parts.append(box(tx, ty, tw, th, "am", "am", AMBR_EC))
parts.append(f'<text x="{tx+tw/2}" y="{ty+74}" text-anchor="middle" font-size="17" '
             f'font-weight="700" fill="{AMBR_TX}">Transformer</text>')
parts.append(f'<text x="{tx+tw/2}" y="{ty+98}" text-anchor="middle" font-size="12.5" '
             f'fill="{AMBR_TX}">~160M params · diffusion forcing</text>')
parts.append(f'<text x="{tx+tw/2}" y="{ty+140}" text-anchor="middle" font-size="13" '
             f'font-style="italic" fill="{AMBR_TX}">AdaLN:  γ(a)·LN(x)+β(a)</text>')

# ── action chunk + embedding ───────────────────────────────────────────────
parts.append(f'<text x="125" y="404" text-anchor="middle" class="sub">action chunk  (Δx, Δθ)</text>')
parts.append(box(40, 414, 170, 56, "am", "am", AMBR_EC))
parts.append(f'<text x="125" y="447" text-anchor="middle" font-size="13.5" '
             f'font-weight="700" fill="{AMBR_TX}">Action Embedding</text>')

# ── predicted latent ───────────────────────────────────────────────────────
px, py, pw, ph = 742, 212, 120, 96
parts.append(box(px, py, pw, ph, "am", "am", AMBR_EC, sw=2.4))
parts.append(f'<text x="{px+pw/2}" y="{py+50}" text-anchor="middle" font-size="20" '
             f'font-style="italic" fill="{AMBR_TX}">z&#770;'
             f'<tspan baseline-shift="sub" font-size="11">t+1</tspan></text>')
parts.append(f'<text x="{px+pw/2}" y="{py+74}" text-anchor="middle" class="sub" '
             f'fill="{AMBR_TX}">predicted latent</text>')

# ── decoder ────────────────────────────────────────────────────────────────
dx, dy, dw, dh = 912, 207, 208, 106
parts.append(box(dx, dy, dw, dh, "gr", "gr", GREN_EC))
parts.append(f'<text x="{dx+dw/2}" y="{dy+44}" text-anchor="middle" font-size="14" '
             f'font-weight="700" fill="{GREN_TX}">Token → RGB Decoder</text>')
parts.append(f'<text x="{dx+dw/2}" y="{dy+66}" text-anchor="middle" class="sub" '
             f'fill="{GREN_TX}">visualization only</text>')
parts.append(f'<text x="{dx+dw/2}" y="190" text-anchor="middle" font-size="11" '
             f'font-style="italic" fill="{GREN_TX}">planner scores in token space — never decodes</text>')

# ── connectors ─────────────────────────────────────────────────────────────
c_cam = "M120,128 L120,158"
c_enc = "M120,230 L120,310"
c_lat = "M292,341 C352,341 360,255 416,255"
c_act = "M212,438 C338,438 392,360 498,360"
c_prd = "M692,255 L740,256"
c_dec = "M862,260 L910,260"
for d in (c_cam, c_enc, c_lat, c_act, c_prd, c_dec):
    parts.append(wire(d))
parts.append(f'<text x="372" y="312" text-anchor="middle" class="sub" '
             f'fill="{AMBR_EC}">γ(a), β(a)</text>')
parts.append(f'<text x="716" y="246" text-anchor="middle" class="sub">denoise</text>')

# ── data-flow pulses ───────────────────────────────────────────────────────
parts.append(pulse(c_enc, "1.6s", "0s", BLUE_EC))
parts.append(pulse(c_lat, "1.7s", "0.3s", BLUE_EC))
parts.append(pulse(c_act, "1.9s", "0.1s", AMBR_EC))
parts.append(pulse(c_prd, "1.2s", "0.2s", AMBR_EC))
parts.append(pulse(c_dec, "1.2s", "0.6s", GREN_EC))

# ── legend ─────────────────────────────────────────────────────────────────
leg = [(BLUE_FC2, BLUE_EC, "Frozen / pretrained", 289),
       (AMBR_FC2, AMBR_EC, "Trained on robot data", 493),
       (GREN_FC2, GREN_EC, "Visualization only", 712)]
for fc, ec, txt, x in leg:
    parts.append(f'<rect x="{x}" y="476" width="18" height="13" rx="3" '
                 f'fill="{fc}" stroke="{ec}" stroke-width="1.4"/>')
    parts.append(f'<text x="{x+25}" y="486" class="lbl">{txt}</text>')

parts.append('</svg>')

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'nanowm_arch.svg')
with open(out, "w") as f:
    f.write("\n".join(parts))
print(f"saved {os.path.abspath(out)}")
