"""Build compare/: our dithering next to epdoptimize 1.3.0 as the paperlesspaper app runs it, for the images in
samples/ and gallery/.

    (cd scripts && npm ci)                          # epdoptimize 1.3.0, pinned in package-lock.json
    python scripts/build_compare.py [--workers 3]   # about a minute per image and worker

Writes compare/index.html, compare/scores.json and, per image, compare/img/<name>.app.png and <name>.ours.png: the
device images (1600x1200 palette PNGs, portrait photos rotated 90 degrees clockwise, as the CLI writes them). The page
renders them through the panel model in the browser (compare/panel.js, a port of spectra6_dotgain/panel.py) and builds
the A/B images there.
Scores: mean S-CIELAB CIEDE2000 at 50 cm (150 ppi) between a simulation and the photo, white-adapted; "model" uses
the panel model, "flat" mixes the inks as photographed pixel by pixel, with no darkening.
"""
import argparse
import hashlib
import html
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectra6_dotgain import DEVICE, ROLES, dither, indices_from_device, render, resize_cover  # noqa: E402
from spectra6_dotgain.prep import srgb_to_linear  # noqa: E402

OUT = ROOT / "compare"
IMG = OUT / "img"
NODE = ROOT / "scripts" / "epdoptimize_app.mjs"

# file in samples/ (sources in samples/NOTICE), title, where the close-up starts (x, y in the photo)
SAMPLES = [
    ("portrait.jpg", "Portrait (AI-generated)", (565, 292)),
    ("parade.jpg", "Parade", (850, 690)),
    ("flags.jpg", "Flags", (520, 380)),
    ("warehouse.jpg", "Warehouse at dusk", (740, 600)),
    ("signpost.jpg", "Signpost in snow", (370, 280)),
    ("grey-ramp.png", "Grey ramp", (150, 100)),
]
# gallery/: epdoptimize's demo images (gallery/NOTICE); the close-up starts in the middle
GALLERY = sorted(p.name for p in (ROOT / "gallery").iterdir() if p.suffix in (".jpg", ".png"))

# ---------------------------------------------------------------------------------------------------- epdoptimize


def epdoptimize_app(ref8):
    """epdoptimize as the app runs it. The app's editor works from a copy of the photo capped at 1280 px on the long
    edge, which Fabric scales back up with image smoothing; resizing to 1280 and back with Pillow's bilinear filter
    approximates that."""
    im = Image.fromarray(ref8)
    W, H = im.size
    s = 1280 / max(W, H)
    small = im.resize((round(W * s), round(H * s)), Image.BILINEAR, reducing_gap=None)
    with tempfile.TemporaryDirectory() as td:
        src, dev = Path(td) / "in.png", Path(td) / "device.png"
        small.resize((W, H), Image.BILINEAR).save(src)
        r = subprocess.run(["node", str(NODE), str(src), str(dev)], capture_output=True, text=True, check=True)
        idx = indices_from_device(np.asarray(Image.open(dev).convert("RGB")))
    return idx, json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------------- score
# S-CIELAB (Zhang & Wandell 1997): XYZ -> opponent channels -> per-channel sum of Gaussians (spreads in degrees)
# -> XYZ -> CIELAB, then CIEDE2000 against the filtered photo. Same maths as the lab's metric.
PPI = 150.0
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
D65 = M_RGB2XYZ @ np.ones(3)
M_XYZ2OPP = np.array([[0.279, 0.720, -0.107], [-0.449, 0.290, -0.077], [0.086, -0.590, 0.501]])
M_OPP2XYZ = np.linalg.inv(M_XYZ2OPP)
SCIELAB_KERNELS = [  # (weights, spreads in degrees) per opponent channel
    ([1.00327, 0.114416, -0.117686], [0.05, 0.225, 7.0]),
    ([0.616725, 0.383275], [0.0685, 0.826]),
    ([0.567885, 0.432115], [0.0920, 0.6451]),
]


def _filter_channel(ch, weights, spreads_deg, ppd, pad):
    p = np.pad(ch, pad, mode="reflect")
    F = np.fft.rfft2(p)
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    tf = np.zeros_like(F, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    for wi, s in zip(w, spreads_deg):
        tf += wi * np.exp(-2 * (np.pi**2) * ((s * ppd) ** 2) * (fx**2 + fy**2))
    return np.fft.irfft2(F * tf, s=p.shape)[pad:-pad, pad:-pad]


def _scielab(lin, ppd):
    opp = (lin @ M_RGB2XYZ.T).astype(np.float32) @ M_XYZ2OPP.T
    pad = int(min(max(opp.shape[:2]) // 2, 3 * 0.826 * ppd))
    out = np.empty_like(opp, dtype=np.float64)
    for c, (w, s) in enumerate(SCIELAB_KERNELS):
        out[..., c] = _filter_channel(opp[..., c].astype(np.float64), w, s, ppd, pad)
    t = np.clip((out @ M_OPP2XYZ.T).astype(np.float32), 0, None) / np.asarray(D65, dtype=np.float32)
    d = 6 / 29
    f = np.where(t > d**3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
                    -1).astype(np.float32)


def delta_e2000(lab1, lab2):
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    Cm = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp, dhp = L2 - L1, C2p - C1p, h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lpm, Cpm, hsum = (L1 + L2) / 2, (C1p + C2p) / 2, h1p + h2p
    hpm = np.where(C1p * C2p == 0, hsum, np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                                                  np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm**7 / (Cpm**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm
    Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    dE = np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
    return dE.astype(np.float32)


def score(sim_lin, ref_lin, distance_mm=500):
    ppd = 2 * distance_mm * np.tan(np.radians(0.5)) / (25.4 / PPI)
    return float(delta_e2000(_scielab(sim_lin, ppd), _scielab(ref_lin, ppd)).mean())


# the inks as photographed (room glare included), for a score that does not use the panel model
PANEL_JSON = json.loads((ROOT / "spectra6_dotgain" / "panel-model.json").read_text())
INKS_PHOTO = np.array([PANEL_JSON["inks_as_photographed"][r] for r in ROLES], np.float32)

# ---------------------------------------------------------------------------------------------------- images


def save_device(idx, path):
    """Palette PNG of the device image (6 colours, 4 bits per pixel), rotated like the CLI's output."""
    im = Image.fromarray(idx, "P")
    im.putpalette(DEVICE.flatten().tolist())
    if im.height > im.width:
        im = im.transpose(Image.Transpose.ROTATE_270)
    im.save(path, optimize=True)


def build_image(job):
    folder, file, title, centre = job
    name = Path(file).stem
    ref8 = np.asarray(resize_cover(Image.open(ROOT / folder / file)))
    H, W = ref8.shape[:2]
    idx = {"ours": dither(ref8)}
    idx["app"], opts = epdoptimize_app(ref8)
    ref_lin = srgb_to_linear(ref8 / 255.0)
    scores = {"model": {}, "flat": {}}
    for v in ("app", "ours"):
        save_device(idx[v], IMG / f"{name}.{v}.png")
        scores["model"][v] = score(render(idx[v]), ref_lin)
        scores["flat"][v] = score(INKS_PHOTO[idx[v]], ref_lin)
    # which version is on top of the A/B image: fixed per image, but not always the same one
    ours_top = hashlib.sha256(name.encode()).digest()[0] % 2 == 0
    print(f"{name[:40]:40s} model: app {scores['model']['app']:5.2f} ours {scores['model']['ours']:5.2f}  "
          f"flat: app {scores['flat']['app']:5.2f} ours {scores['flat']['ours']:5.2f}  {opts['imageKind']} "
          f"{opts['options'].get('errorDiffusionMatrix')}", flush=True)
    return {"name": name, "folder": folder, "file": file, "title": title, "w": W, "h": H,
            "cx": centre[0] if centre else W // 2, "cy": centre[1] if centre else H // 2,
            "scores": scores, "epdoptimize": opts, "ab": ("ours", "app") if ours_top else ("app", "ours")}


# ---------------------------------------------------------------------------------------------------- page
COLS = [("photo", "Photo"), ("app", "paperlesspaper app, re-created"), ("ours", "spectra6-dotgain")]
KINDS = {"photo": "photo", "lowContrastPhoto": "low-contrast photo", "highContrastPhoto": "high-contrast photo",
         "flatIllustration": "flat illustration", "lineArt": "line art", "textOrUi": "text or UI",
         "pixelArt": "pixel art", "unknown": "unknown"}
KERNELS = {"floydSteinberg": "Floyd-Steinberg", "stucki": "Stucki", "atkinson": "Atkinson", "sierra3": "Sierra"}


def section(im):
    n, W, H = im["name"], im["w"], im["h"]
    e = html.escape
    ep = im["epdoptimize"]
    kind = KINDS.get(ep["imageKind"], ep["imageKind"])
    conf = f", confidence {ep['confidence']:.2f}" if ep.get("confidence") is not None else ""
    o = ep["options"]
    how = (KERNELS.get(o.get("errorDiffusionMatrix"), o.get("errorDiffusionMatrix"))
           if o.get("ditheringType") == "errorDiffusion" else "no dithering, nearest colour")
    if o.get("colorMatching", "rgb") != "rgb":
        how += ", colours matched in " + o["colorMatching"].capitalize()
    figs = []
    for key, label in COLS:
        if key == "photo":
            sub = f' <a class="m" href="../{im["folder"]}/{e(im["file"])}">file</a>'
        elif key == "app":
            sub = f'<br><span class="m">epdoptimize: {e(kind)}{conf}; {e(how)}</span>'
        else:
            sub = ""
        figs.append(f'<figure class="{key}"><div class="view"><canvas class="ov" width="{W // 2}" height="{H // 2}" '
                    f'style="aspect-ratio:{W}/{H}"></canvas><canvas class="inset"></canvas><div class="hl"></div></div>'
                    f'<figcaption>{e(label)}{sub}</figcaption></figure>')
    names = dict(COLS)
    top, bottom = im["ab"]
    return f"""
<section id="{e(n)}" data-name="{e(n)}" data-w="{W}" data-h="{H}" data-cx="{im['cx']}" data-cy="{im['cy']}"
  data-photo="../{im['folder']}/{e(im['file'])}" data-top="{top}">
  <h3>{e(im["title"])}</h3>
  <div class="row">{"".join(figs)}</div>
  <div class="ab m"><button type="button" class="abbtn">A/B image</button> to send to a real frame: the middle of the
  photo twice, one version above the other. <details><summary>Which is which</summary>top: {names[top]},
  bottom: {names[bottom]}</details></div>
</section>"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spectra6-dotgain comparison</title>
<style>
:root { --bg: #fafaf8; --fg: #1c1c1e; --m: #62626a; --line: #dcdcd6; --hi: #d9480f; color-scheme: light dark; }
@media (prefers-color-scheme: dark) { :root { --bg: #161618; --fg: #e6e6e8; --m: #9a9aa2; --line: #36363c; --hi: #ff8a4c; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; line-height: 1.3; margin: 8px 0 12px; }
h2 { font-size: 20px; margin: 32px 0 4px; }
h3 { font-size: 17px; margin: 0 0 10px; overflow-wrap: anywhere; }
a { color: inherit; }
.m { color: var(--m); font-size: 13px; font-weight: normal; }
.intro { max-width: 72ch; }
.bar { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line);
  display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.bar span { white-space: nowrap; }
.off #zooms { display: none; }
section { padding: 20px 0 8px; border-bottom: 1px solid var(--line); }
.row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items: start; }
figure { margin: 0; }
figcaption { margin-top: 4px; font-size: 14px; }
.view { position: relative; cursor: crosshair; line-height: 0; }
.ov { width: 100%; height: auto; display: block; background: var(--line); }
.inset { position: absolute; left: 0; bottom: 0; width: 40%; aspect-ratio: 1; image-rendering: pixelated;
  border: 1px solid rgba(255,255,255,.9); box-shadow: 0 0 0 1px rgba(0,0,0,.45); pointer-events: none; }
.hl { position: absolute; display: none; outline: 1px solid rgba(255,255,255,.75); box-shadow: 0 0 0 1px rgba(0,0,0,.35);
  pointer-events: none; }
.hover .hl { display: block; }
.off .inset, .off .hl { display: none !important; }
button { font: inherit; font-size: 13px; color: var(--fg); background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 3px 9px; cursor: pointer; }
button[aria-pressed=true] { border-color: var(--hi); color: var(--hi); }
.ab { margin: 10px 0 0; }
details { display: inline-block; margin-left: 8px; }
summary { cursor: pointer; }
#err { display: none; border-left: 3px solid var(--hi); padding-left: 10px; }
@media (max-width: 700px) {
  .row { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .row > .photo { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<main>
<h1>The paperlesspaper app vs spectra6-dotgain</h1>
<p class="intro">Images dithered for the OpenPaper L (1600&times;1200) two ways: by epdoptimize 1.3.0 the way the
paperlesspaper app runs it, re-created from the app's source, and by spectra6-dotgain. Both are simulated as the panel
would show them, with a model measured on a real panel in which darker pixels bleed into lighter neighbours. Our
dithering is designed for that model, so it has the advantage here; the
<a href="https://github.com/marinsokol5/spectra6-dotgain#results">README</a> has the tests on a real frame.</p>
<p class="m intro">The square in each image's corner is a close-up; click anywhere on an image to move it, for all
three images of the row. Panel white is shown as screen white. The model's black is a little purple-blue in both
simulations.</p>
<p id="err" class="intro">This page reads its images' pixels, which browsers block for files opened from disk. Open it
from a web server instead, for example <code>python3 -m http.server</code> in the repository folder.</p>
<script>if (location.protocol === "file:") document.getElementById("err").style.display = "block";</script>
<div class="bar"><button type="button" id="closeup" aria-pressed="true">Close-up</button>
<span id="zooms"><button type="button" data-z="2">2&times;</button> <button type="button" data-z="3">3&times;</button>
<button type="button" data-z="4">4&times;</button> <span class="m">screen px per panel px</span></span></div>
<h2>paperlesspaper's sample images</h2>
__SAMPLES__
<h2>epdoptimize's demo images</h2>
<p class="m intro">All of them except four that show the same picture as a sample above. Under each app image: how
epdoptimize's <code>suggestCanvasProcessingOptions</code> classifies the image, and how it then dithers.</p>
__GALLERY__
<p class="m">Sample images: paperlesspaper's (samples/NOTICE). Demo images: from epdoptimize (gallery/NOTICE). Both
under their own terms.</p>
</main>
<script type="module">
import { abDevice, loadModel, render, srgb8, LINEAR, indices, pngRGB } from "./panel.js";

const f32 = Math.fround;
const q = new URLSearchParams(location.search);
let closeup = q.get("closeup") !== "off";
let zoom = [2, 3, 4].includes(+q.get("zoom")) ? +q.get("zoom") : 3;

function showState() {
  document.body.classList.toggle("off", !closeup);
  document.getElementById("closeup").setAttribute("aria-pressed", String(closeup));
  document.querySelectorAll("[data-z]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.z === zoom)));
}
function setState() {
  showState();
  history.replaceState(null, "", `?closeup=${closeup ? "on" : "off"}&zoom=${zoom}`);
  rows.forEach(r => r.drawInsets());
}
document.getElementById("closeup").addEventListener("click", () => { closeup = !closeup; setState(); });
document.querySelectorAll("[data-z]").forEach(b => b.addEventListener("click", () => {
  zoom = +b.dataset.z;
  setState();
}));
showState();

const image = src => new Promise((ok, fail) => {
  const im = new Image();
  im.onload = () => ok(im);
  im.onerror = () => fail(new Error(src));
  im.src = src;
});
function pixels(im) {
  const c = document.createElement("canvas");
  c.width = im.naturalWidth;
  c.height = im.naturalHeight;
  const x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(im, 0, 0);
  return x.getImageData(0, 0, c.width, c.height).data;
}
// device image (panel orientation) -> index map in the photo's orientation
function deviceIndices(im, W, H) {
  const dev = indices(pixels(im)), dw = im.naturalWidth;
  if (W === dw) return dev;
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = dev[x * dw + (dw - 1 - y)];
  return out;
}
// 2x2 panel pixels -> 1 overview pixel, averaged in linear light
function overview(canvas, lin, W, H) {
  const w = W >> 1, h = H >> 1, out = new ImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
    const a = (2 * y * W + 2 * x) * 3 + c, b = a + W * 3;
    out.data[(y * w + x) * 4 + c] = srgb8(f32(f32(f32(f32(lin[a] + lin[a + 3]) + lin[b]) + lin[b + 3]) * 0.25));
  }
  for (let i = 3; i < out.data.length; i += 4) out.data[i] = 255;
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").putImageData(out, 0, 0);
}

const model = fetch("../spectra6_dotgain/panel-model.json").then(r => r.json()).then(loadModel);

class Row {
  constructor(sec) {
    this.sec = sec;
    this.name = sec.dataset.name;
    this.W = +sec.dataset.w;
    this.H = +sec.dataset.h;
    this.cx = +sec.dataset.cx;
    this.cy = +sec.dataset.cy;
    this.views = [...sec.querySelectorAll(".view")];
    this.data = null;
    this.views.forEach(v => {
      v.addEventListener("click", ev => { [this.cx, this.cy] = this.at(v, ev); this.drawInsets(); this.hover(v, ev); });
      v.addEventListener("mousemove", ev => this.hover(v, ev));
      v.addEventListener("mouseleave", () => sec.classList.remove("hover"));
    });
    sec.querySelector(".abbtn").addEventListener("click", () => this.download());
  }
  at(v, ev) {
    const r = v.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * this.W, (ev.clientY - r.top) / r.height * this.H];
  }
  region(cx, cy) {
    const css = this.views[0].clientWidth * 0.4, s = Math.max(8, Math.min(Math.round(css / zoom), this.W, this.H));
    const x0 = Math.min(Math.max(Math.round(cx - s / 2), 0), this.W - s);
    const y0 = Math.min(Math.max(Math.round(cy - s / 2), 0), this.H - s);
    return [x0, y0, s];
  }
  hover(v, ev) {
    if (!closeup) return;
    const [x0, y0, s] = this.region(...this.at(v, ev));
    this.sec.querySelectorAll(".hl").forEach(h => Object.assign(h.style, {
      left: `${100 * x0 / this.W}%`, top: `${100 * y0 / this.H}%`, width: `${100 * s / this.W}%`,
      height: `${100 * s / this.H}%`,
    }));
    this.sec.classList.add("hover");
  }
  async load() {
    if (this.data) return this.data;
    const sec = this.sec;
    const [photo, app, ours] = await Promise.all([image(sec.dataset.photo), image(`img/${this.name}.app.png`),
      image(`img/${this.name}.ours.png`)]);
    this.data = { photo, idx: { app: deviceIndices(app, this.W, this.H), ours: deviceIndices(ours, this.W, this.H) } };
    return this.data;
  }
  async draw() {
    const { photo, idx } = await this.load(), m = await model, ovs = this.sec.querySelectorAll(".ov");
    const px = pixels(photo), lin = new Float32Array(this.W * this.H * 3);
    for (let i = 0; i < this.W * this.H; i++) for (let c = 0; c < 3; c++) lin[i * 3 + c] = LINEAR[px[i * 4 + c]];
    overview(ovs[0], lin, this.W, this.H);
    await new Promise(r => setTimeout(r));
    overview(ovs[1], render(idx.app, this.W, this.H, m), this.W, this.H);
    await new Promise(r => setTimeout(r));
    overview(ovs[2], render(idx.ours, this.W, this.H, m), this.W, this.H);
    this.drawn = true;
    await this.drawInsets();
  }
  async drawInsets() {
    if (!this.data || !closeup) return;
    const m = await model, { photo, idx } = this.data, [x0, y0, s] = this.region(this.cx, this.cy);
    const insets = this.sec.querySelectorAll(".inset");
    insets.forEach(c => { c.width = s; c.height = s; });
    insets[0].getContext("2d").drawImage(photo, x0, y0, s, s, 0, 0, s, s);
    ["app", "ours"].forEach((v, k) => {
      // the region plus a 1 px border (clamped at the image's edges), rendered, border dropped
      const n = s + 2, patch = new Uint8Array(n * n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
        const yy = Math.min(Math.max(y0 + y - 1, 0), this.H - 1), xx = Math.min(Math.max(x0 + x - 1, 0), this.W - 1);
        patch[y * n + x] = idx[v][yy * this.W + xx];
      }
      const lin = render(patch, n, n, m), out = new ImageData(s, s);
      for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) for (let c = 0; c < 4; c++) {
        out.data[(y * s + x) * 4 + c] = c === 3 ? 255 : srgb8(lin[((y + 1) * n + x + 1) * 3 + c]);
      }
      insets[k + 1].getContext("2d").putImageData(out, 0, 0);
    });
  }
  release() {
    this.data = null;
    this.drawn = false;
    this.sec.querySelectorAll("canvas").forEach(c => { c.width = 0; c.height = 0; });
  }
  async download() {
    const { idx } = await this.load();
    const [top, bottom] = this.sec.dataset.top === "ours" ? ["ours", "app"] : ["app", "ours"];
    const { rgb, w, h } = abDevice(idx[top], idx[bottom], this.W, this.H);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await pngRGB(rgb, w, h));
    a.download = `${this.name}.ab.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }
}

const rows = [...document.querySelectorAll("section[data-name]")].map(s => new Row(s));
const bySec = new Map(rows.map(r => [r.sec, r]));
// draw rows near the screen one at a time, and drop the decoded images of rows far away
let queue = [], busy = false;
async function pump() {
  if (busy) return;
  busy = true;
  while (queue.length) {
    const r = queue.shift();
    if (r.near && !r.drawn) {
      try { await r.draw(); } catch (e) { document.getElementById("err").style.display = "block"; queue = []; }
    }
    await new Promise(r => setTimeout(r));
  }
  busy = false;
}
const near = new IntersectionObserver(es => es.forEach(e => {
  const r = bySec.get(e.target);
  r.near = e.isIntersecting;
  if (r.near && !r.drawn) { queue.push(r); pump(); }
}), { rootMargin: "100% 0px" });
const far = new IntersectionObserver(es => es.forEach(e => {
  const r = bySec.get(e.target);
  if (!e.isIntersecting && r.data) r.release();
}), { rootMargin: "400% 0px" });
rows.forEach(r => { near.observe(r.sec); far.observe(r.sec); });
let t;
addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => rows.forEach(r => r.drawInsets()), 200); });
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    jobs = [("samples", f, t, c) for f, t, c in SAMPLES] + [("gallery", f, Path(f).stem, None) for f in GALLERY]
    with ProcessPoolExecutor(a.workers) as ex:
        images = list(ex.map(build_image, jobs))
    groups = {"samples": [im for im in images if im["folder"] == "samples"],
              "gallery": [im for im in images if im["folder"] == "gallery"]}
    mean = {g: {k: {v: round(float(np.mean([im["scores"][k][v] for im in ims])), 2) for v in ("app", "ours")}
                for k in ("model", "flat")} for g, ims in groups.items()}
    per = {im["name"]: {k: {v: round(s, 2) for v, s in im["scores"][k].items()} for k in ("model", "flat")}
           | {"epdoptimize": im["epdoptimize"]} for im in images}
    (OUT / "scores.json").write_text(json.dumps({"mean": mean, **per}, indent=1) + "\n")
    page = PAGE.replace("__SAMPLES__", "".join(section(im) for im in groups["samples"]))
    (OUT / "index.html").write_text(page.replace("__GALLERY__", "".join(section(im) for im in groups["gallery"])))
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
