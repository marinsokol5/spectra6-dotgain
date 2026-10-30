"""Build compare/: our dithering next to epdoptimize 1.3.0 as the paperlesspaper app runs it, for the images in
samples/ and gallery/.

    (cd scripts && npm ci)                          # epdoptimize 1.3.0, pinned in package-lock.json
    python scripts/build_compare.py [--workers 3]   # about a minute per image and worker

Writes compare/index.html, compare/scores.json and, per image, compare/thumb/<name>.jpg (the gallery's thumbnail)
and compare/img/<name>.app.png and <name>.ours.png: the device images (1600x1200 palette PNGs, portrait photos rotated
90 degrees clockwise, as the CLI writes them). The page renders them through the panel model in the browser
(compare/panel.js, a port of spectra6_dotgain/panel.py).
Scores: mean S-CIELAB CIEDE2000 at 50 cm (150 ppi) between a simulation and the photo, white-adapted; "model" uses
the panel model, "flat" mixes the inks as photographed pixel by pixel, with no darkening.
"""
import argparse
import html
import json
import subprocess
import sys
import tempfile
from urllib.parse import quote
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectra6_dotgain import DEVICE, ROLES, dither, indices_from_device, render, resize_cover  # noqa: E402
from spectra6_dotgain.prep import PANEL, srgb_to_linear, to_srgb  # noqa: E402

OUT = ROOT / "compare"
IMG = OUT / "img"
THUMB = OUT / "thumb"
NODE = ROOT / "scripts" / "epdoptimize_app.mjs"

# file in samples/ (sources in samples/NOTICE), title
SAMPLES = [
    ("portrait.jpg", "Portrait (AI-generated)"),
    ("parade.jpg", "Parade"),
    ("flags.jpg", "Flags"),
    ("warehouse.jpg", "Warehouse at dusk"),
    ("signpost.jpg", "Signpost in snow"),
    ("grey-ramp.png", "Grey ramp"),
]
# gallery/: epdoptimize's demo images (gallery/NOTICE)
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


def meta(folder, file, title):
    """What the page needs to know about an image without dithering it."""
    name = Path(file).stem
    im = to_srgb(Image.open(ROOT / folder / file))
    W, H = PANEL[::-1] if im.height > im.width else PANEL  # as resize_cover: portrait photos stay portrait
    return {"name": name, "folder": folder, "file": file, "title": title, "w": W, "h": H}


def save_thumb(job):
    """The gallery's thumbnail: the photo as it is dithered, a quarter of the size."""
    folder, file, _ = job
    im = resize_cover(Image.open(ROOT / folder / file))
    im.resize((im.width // 4, im.height // 4), Image.LANCZOS).save(THUMB / f"{Path(file).stem}.jpg", quality=80)


def build_image(job):
    folder, file, title = job
    name = Path(file).stem
    ref8 = np.asarray(resize_cover(Image.open(ROOT / folder / file)))
    idx = {"ours": dither(ref8)}
    idx["app"], opts = epdoptimize_app(ref8)
    ref_lin = srgb_to_linear(ref8 / 255.0)
    scores = {"model": {}, "flat": {}}
    for v in ("app", "ours"):
        save_device(idx[v], IMG / f"{name}.{v}.png")
        scores["model"][v] = score(render(idx[v]), ref_lin)
        scores["flat"][v] = score(INKS_PHOTO[idx[v]], ref_lin)
    print(f"{name[:40]:40s} model: app {scores['model']['app']:5.2f} ours {scores['model']['ours']:5.2f}  "
          f"flat: app {scores['flat']['app']:5.2f} ours {scores['flat']['ours']:5.2f}  {opts['imageKind']} "
          f"{opts['options'].get('errorDiffusionMatrix')}", flush=True)
    return meta(folder, file, title) | {"scores": scores, "epdoptimize": opts}


# ---------------------------------------------------------------------------------------------------- page
COLS = [("photo", "Photo"), ("app", "paperlesspaper app, re-created"), ("ours", "spectra6-dotgain")]
KINDS = {"photo": "a photo", "lowContrastPhoto": "a low-contrast photo", "highContrastPhoto": "a high-contrast photo",
         "flatIllustration": "a flat illustration", "lineArt": "line art", "textOrUi": "text or UI",
         "pixelArt": "pixel art", "unknown": "unknown"}
KERNELS = {"floydSteinberg": "Floyd-Steinberg", "stucki": "Stucki", "atkinson": "Atkinson", "sierra3": "Sierra"}
CODE = "https://github.com/marinsokol5/spectra6-dotgain/blob/main/"


def suggestion(ep):
    """What epdoptimize's suggestCanvasProcessingOptions made of the image, in a sentence."""
    kind = KINDS.get(ep["imageKind"], ep["imageKind"])
    conf = f" (confidence {ep['confidence']:.2f})" if ep.get("confidence") is not None else ""
    o = ep["options"]
    lab = o.get("colorMatching", "rgb") != "rgb"
    if o.get("ditheringType") == "errorDiffusion":
        how = "dithers it with " + KERNELS.get(o.get("errorDiffusionMatrix"), o.get("errorDiffusionMatrix"))
        how += ", matching colours in Lab" if lab else ""
    else:
        how = "does not dither: each pixel gets the nearest colour" + (" in Lab" if lab else "")
    return f"epdoptimize classifies it as {kind}{conf} and {how}."


def section(im):
    n, W, H = im["name"], im["w"], im["h"]
    e = html.escape
    src = f"../{im['folder']}/{quote(im['file'])}"
    figs = "".join(f'<figure class="{key}"><figcaption>{e(label)}</figcaption><div class="view"><canvas class="ov" '
                   f'width="{W // 2}" height="{H // 2}" style="aspect-ratio:{W}/{H}"></canvas><div class="hl"></div>'
                   f'</div><canvas class="zoom"></canvas></figure>' for key, label in COLS)
    return f"""
<section hidden data-name="{e(n)}" data-w="{W}" data-h="{H}" data-photo="{e(src)}">
  <h2>{e(im["title"])}</h2>
  <p class="m">{e(suggestion(im["epdoptimize"]))} <a href="{e(src)}">Photo file</a></p>
  <div class="row">{figs}</div>
</section>"""


def tile(im):
    e = html.escape
    w, h = im["w"] // 4, im["h"] // 4
    return (f'<a class="tile" href="#{e(quote(im["name"]))}" data-name="{e(im["name"])}"><img '
            f'src="thumb/{e(quote(im["name"]))}.jpg" width="{w}" height="{h}" alt="" loading="lazy">'
            f'<span>{e(im["title"])}</span></a>')


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
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; line-height: 1.3; margin: 8px 0 12px; }
h2 { font-size: 19px; margin: 32px 0 8px; overflow-wrap: anywhere; }
section h2 { margin: 20px 0 2px; }
a { color: inherit; }
.m { color: var(--m); font-size: 13px; font-weight: normal; }
.intro { max-width: 72ch; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px 12px; }
.tile { text-decoration: none; display: block; }
.tile img { display: block; width: 100%; height: auto; aspect-ratio: 4 / 3; object-fit: cover; background: var(--line); }
.tile span { display: block; margin-top: 4px; font-size: 13px; line-height: 1.35; overflow-wrap: anywhere; }
.tile:hover span { text-decoration: underline; }
.bar { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line);
  display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
.bar .end { margin-left: auto; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.bar span { white-space: nowrap; }
.off #zooms { display: none; }
.row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px 12px; margin-top: 12px; }
figure { margin: 0; display: grid; grid-row: span 3; grid-template-rows: subgrid; }
figcaption { align-self: end; font-size: 14px; }
.view { position: relative; cursor: grab; line-height: 0; }
.view:active { cursor: grabbing; }
.ov { width: 100%; height: auto; display: block; background: var(--line); }
.hl { position: absolute; outline: 2px solid #fff; box-shadow: 0 0 0 3px rgba(0,0,0,.5); pointer-events: none; }
.zoom { display: block; width: 100%; aspect-ratio: 4 / 3; image-rendering: pixelated; background: var(--line); }
.off .zoom, .off .hl { display: none; }
button { font: inherit; font-size: 13px; color: var(--fg); background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 3px 9px; cursor: pointer; }
button[aria-pressed=true] { border-color: var(--hi); color: var(--hi); }
#err { display: none; border-left: 3px solid var(--hi); padding-left: 10px; }
@media (max-width: 700px) {
  .row { grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 8px; }
  .row > .photo { grid-column: 1 / -1; }
  .row > .photo .zoom { margin-bottom: 10px; }
}
</style>
</head>
<body>
<main>
<p id="err" class="intro">This page reads its images' pixels, which browsers block for files opened from disk. Open it
from a web server instead, for example <code>python3 -m http.server</code> in the repository folder.</p>
<script>if (location.protocol === "file:") document.getElementById("err").style.display = "block";</script>
<div id="index">
<h1>spectra6-dotgain vs the paperlesspaper app</h1>
<p class="intro">On an E Ink Spectra 6 panel, a light pixel next to darker pixels looks darker than its ink. We measured
this on a paperlesspaper OpenPaper L (13.3", 1600&times;1200): with 1, 2, 3 or 4 darker neighbours, 20, 27, 33 or 51 %
of a pixel takes their colour. Yellow bleeds into white almost as much as black does, and green in fine patterns looks
yellow-green.</p>
<p class="intro">spectra6-dotgain dithers for this. It maps the photo to what the inks can show, then runs error
diffusion that tries every ink for each pixel, predicts with the panel model how that pixel and its neighbours will
look, and keeps the closest. Code: <a href="__CODE__spectra6_dotgain/target.py">tone and colour</a>,
<a href="__CODE__spectra6_dotgain/mbed.py">dithering</a>, <a href="__CODE__spectra6_dotgain/panel.py">panel model</a>.
The <a href="__CODE__README.md">README</a> has the measurements, the blind tests on a real frame and the limits.</p>
<p class="intro">Click an image to see it three ways: the photo, epdoptimize 1.3.0 as the paperlesspaper app runs it
(re-created from the app's source), and spectra6-dotgain. Both dithered versions are simulated with the panel model.
Our dithering is designed for that model, so the simulation favours us.</p>
<h2>paperlesspaper's sample images</h2>
<div class="grid">__SAMPLE_TILES__</div>
<h2>epdoptimize's demo images</h2>
<p class="m intro">All of them except four that show the same picture as a sample above.</p>
<div class="grid">__GALLERY_TILES__</div>
<p class="m">Sample images: paperlesspaper's (samples/NOTICE). Demo images: from epdoptimize (gallery/NOTICE). Both
under their own terms.</p>
</div>
<div id="detail" hidden>
<div class="bar"><a href="#" id="back">&larr; All images</a>
<span><a href="#" id="prev">&larr; Previous</a> <span id="pos" class="m"></span> <a href="#" id="next">Next &rarr;</a></span>
<span class="end"><button type="button" id="closeup" aria-pressed="true">Close-up</button>
<span id="zooms"><button type="button" data-z="1">1&times;</button> <button type="button" data-z="2">2&times;</button>
<button type="button" data-z="3">3&times;</button> <button type="button" data-z="4">4&times;</button>
<span class="m">screen px per panel px</span></span></span></div>
__SECTIONS__
<p class="m intro">Under each image is a close-up of the white box on it. Drag the box, or click anywhere on an image,
to move it in all three. Panel white is shown as screen white. The model's black is a little purple-blue in both
simulations. Keys: &larr; and &rarr; for the previous and next image, Esc for all images.</p>
</div>
</main>
<script type="module">
import { loadModel, render, srgb8, LINEAR, indices } from "./panel.js";

const f32 = Math.fround;
const q = new URLSearchParams(location.search);
let closeup = q.get("closeup") !== "off";
let zoom = [1, 2, 3, 4].includes(+q.get("zoom")) ? +q.get("zoom") : 2;
let current = null;

function showState() {
  document.body.classList.toggle("off", !closeup);
  document.getElementById("closeup").setAttribute("aria-pressed", String(closeup));
  document.querySelectorAll("[data-z]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.z === zoom)));
}
function setState() {
  showState();
  history.replaceState(null, "", `?closeup=${closeup ? "on" : "off"}&zoom=${zoom}${location.hash}`);
  current?.drawZoom();
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
const tick = () => new Promise(r => setTimeout(r));

class Row {
  constructor(sec) {
    this.sec = sec;
    this.name = sec.dataset.name;
    this.title = sec.querySelector("h2").textContent;
    this.W = +sec.dataset.w;
    this.H = +sec.dataset.h;
    [this.cx, this.cy] = [0, this.H];  // the box starts in the bottom-left corner
    this.views = [...sec.querySelectorAll(".view")];
    this.data = null;
    this.gen = 0;  // bumped on every draw and release, so a draw that is no longer wanted stops
    this.views.forEach(v => {
      // mouse and pen drag; a touch scrolls the page instead, and a tap still moves the box (click)
      v.addEventListener("pointerdown", ev => {
        if (ev.pointerType === "touch" || !closeup) return;
        v.setPointerCapture(ev.pointerId);
        this.dragging = v;
        this.moveTo(v, ev);
      });
      v.addEventListener("pointermove", ev => { if (this.dragging === v) this.moveTo(v, ev); });
      v.addEventListener("pointerup", () => { this.dragging = null; });
      v.addEventListener("click", ev => { if (closeup) this.moveTo(v, ev); });
    });
  }
  at(v, ev) {
    const r = v.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width * this.W, (ev.clientY - r.top) / r.height * this.H];
  }
  // the box in panel pixels: 4:3, as wide as the close-up under the image divided by the zoom
  region(cx, cy) {
    const w = Math.max(8, Math.min(Math.round(this.views[0].clientWidth / zoom), this.W));
    const h = Math.min(Math.round(w * 3 / 4), this.H);
    const x0 = Math.min(Math.max(Math.round(cx - w / 2), 0), this.W - w);
    const y0 = Math.min(Math.max(Math.round(cy - h / 2), 0), this.H - h);
    return [x0, y0, w, h];
  }
  moveTo(v, ev) {
    [this.cx, this.cy] = this.at(v, ev);
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.drawZoom(); });
  }
  mark() {
    const [x0, y0, w, h] = this.region(this.cx, this.cy);
    this.sec.querySelectorAll(".hl").forEach(b => Object.assign(b.style, {
      left: `${100 * x0 / this.W}%`, top: `${100 * y0 / this.H}%`, width: `${100 * w / this.W}%`,
      height: `${100 * h / this.H}%`,
    }));
  }
  async draw() {
    const gen = ++this.gen, sec = this.sec;
    this.mark();
    const [photo, app, ours, m] = await Promise.all([image(sec.dataset.photo), image(`img/${this.name}.app.png`),
      image(`img/${this.name}.ours.png`), model]);
    if (gen !== this.gen) return;
    this.data = { photo, idx: { app: deviceIndices(app, this.W, this.H), ours: deviceIndices(ours, this.W, this.H) } };
    const ovs = sec.querySelectorAll(".ov"), px = pixels(photo), lin = new Float32Array(this.W * this.H * 3);
    for (let i = 0; i < this.W * this.H; i++) for (let c = 0; c < 3; c++) lin[i * 3 + c] = LINEAR[px[i * 4 + c]];
    overview(ovs[0], lin, this.W, this.H);
    await this.drawZoom();
    for (const [k, v] of [[1, "app"], [2, "ours"]]) {
      await tick();
      if (gen !== this.gen) return;
      overview(ovs[k], render(this.data.idx[v], this.W, this.H, m), this.W, this.H);
    }
  }
  async drawZoom() {
    this.mark();
    if (!this.data || !closeup) return;
    const m = await model, { photo, idx } = this.data, [x0, y0, w, h] = this.region(this.cx, this.cy);
    const zooms = this.sec.querySelectorAll(".zoom");
    zooms.forEach(c => { c.width = w; c.height = h; });
    zooms[0].getContext("2d").drawImage(photo, x0, y0, w, h, 0, 0, w, h);
    ["app", "ours"].forEach((v, k) => {
      // the region plus a 1 px border (clamped at the image's edges), rendered, border dropped
      const nw = w + 2, nh = h + 2, patch = new Uint8Array(nw * nh);
      for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
        const yy = Math.min(Math.max(y0 + y - 1, 0), this.H - 1), xx = Math.min(Math.max(x0 + x - 1, 0), this.W - 1);
        patch[y * nw + x] = idx[v][yy * this.W + xx];
      }
      const lin = render(patch, nw, nh, m), out = new ImageData(w, h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) {
        out.data[(y * w + x) * 4 + c] = c === 3 ? 255 : srgb8(lin[((y + 1) * nw + x + 1) * 3 + c]);
      }
      zooms[k + 1].getContext("2d").putImageData(out, 0, 0);
    });
  }
  release() {
    this.gen++;
    this.data = null;
    this.sec.querySelectorAll("canvas").forEach(c => { c.width = 0; c.height = 0; });
  }
}

// #name opens that image; no hash shows the gallery
const rows = [...document.querySelectorAll("section[data-name]")].map(s => new Row(s));
const byName = new Map(rows.map(r => [r.name, r]));
const [index, detail, err] = ["index", "detail", "err"].map(id => document.getElementById(id));
const TITLE = document.title;
const hashOf = r => `#${encodeURIComponent(r.name)}`;
function link(a, r) {
  a.style.visibility = r ? "" : "hidden";
  if (r) a.href = hashOf(r);
}
function route() {
  let name = "";
  try { name = decodeURIComponent(location.hash.slice(1)); } catch {}
  const r = byName.get(name) || null, last = current;
  if (last && last !== r) { last.sec.hidden = true; last.release(); }
  current = r;
  index.hidden = !!r;
  detail.hidden = !r;
  if (r) {
    const i = rows.indexOf(r);
    r.sec.hidden = false;
    document.getElementById("pos").textContent = `${i + 1} / ${rows.length}`;
    link(document.getElementById("prev"), rows[i - 1]);
    link(document.getElementById("next"), rows[i + 1]);
    document.title = `${r.title} - spectra6-dotgain`;
    scrollTo(0, 0);
    if (r !== last) r.draw().catch(() => { err.style.display = "block"; });
  } else {
    document.title = TITLE;
    if (last) document.querySelector(`.tile[data-name="${CSS.escape(last.name)}"]`)?.scrollIntoView({ block: "center" });
  }
}
history.scrollRestoration = "manual";
addEventListener("hashchange", route);
route();
addEventListener("keydown", ev => {
  if (!current || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const i = rows.indexOf(current);
  if (ev.key === "ArrowLeft" && i > 0) location.hash = hashOf(rows[i - 1]);
  else if (ev.key === "ArrowRight" && i < rows.length - 1) location.hash = hashOf(rows[i + 1]);
  else if (ev.key === "Escape") location.hash = "";
});
let t;
addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => current?.drawZoom(), 200); });
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--page", action="store_true",
                    help="only rewrite index.html, from scores.json and the device PNGs already in compare/img")
    a = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    THUMB.mkdir(exist_ok=True)
    jobs = [("samples", f, t) for f, t in SAMPLES] + [("gallery", f, Path(f).stem) for f in GALLERY]
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(save_thumb, jobs))
    if a.page:
        per = json.loads((OUT / "scores.json").read_text())
        images = [meta(*j) | {"epdoptimize": per[Path(j[1]).stem]["epdoptimize"]} for j in jobs]
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            images = list(ex.map(build_image, jobs))
    groups = {"samples": [im for im in images if im["folder"] == "samples"],
              "gallery": [im for im in images if im["folder"] == "gallery"]}
    if not a.page:
        mean = {g: {k: {v: round(float(np.mean([im["scores"][k][v] for im in ims])), 2) for v in ("app", "ours")}
                    for k in ("model", "flat")} for g, ims in groups.items()}
        per = {im["name"]: {k: {v: round(s, 2) for v, s in im["scores"][k].items()} for k in ("model", "flat")}
               | {"epdoptimize": im["epdoptimize"]} for im in images}
        (OUT / "scores.json").write_text(json.dumps({"mean": mean, **per}, indent=1) + "\n")
    page = (PAGE.replace("__CODE__", CODE)
            .replace("__SAMPLE_TILES__", "".join(tile(im) for im in groups["samples"]))
            .replace("__GALLERY_TILES__", "".join(tile(im) for im in groups["gallery"]))
            .replace("__SECTIONS__", "".join(section(im) for im in images)))
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
