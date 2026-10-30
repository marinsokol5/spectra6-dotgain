"""Build compare/: our dithering next to epdoptimize 1.3.0 as the paperlesspaper app runs it, on the images in
samples/, both seen through the measured panel model (spectra6_dither/panel-model.json).

    (cd scripts && npm ci)              # epdoptimize 1.3.0, pinned in package-lock.json
    python scripts/build_compare.py     # under a minute per image

Writes compare/index.html, compare/scores.json and compare/img/: per image a stacked A/B device PNG (1600x1200,
6 device colours), half-size views (2x2 panel pixels averaged in linear light) and 1:1 crops of the simulation.
Scores: mean S-CIELAB CIEDE2000 at 50 cm (150 ppi) between a simulation and the photo, white-adapted; "model" uses
the panel model, "flat" mixes the inks as photographed pixel by pixel, with no darkening.
"""
import hashlib
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectra6_dither import (ROLES, device_image, dither, indices_from_device, linear_to_srgb,  # noqa: E402
                             render, resize_cover)
from spectra6_dither.prep import srgb_to_linear  # noqa: E402

OUT = ROOT / "compare"
IMG = OUT / "img"
NODE = ROOT / "scripts" / "epdoptimize_app.mjs"
CROP = (200, 150)  # panel pixels (w, h) of a landscape crop; swapped for portrait images

# file in samples/, title, upstream file (paperlesspaper/sample-images), crops: label and centre (x, y) in the photo
SAMPLES = [
    ("portrait.jpg", "Portrait (AI-generated)",
     "AI generated/Vercity_Dreamgaze_1960s_hippie_girl._technicolor_old_movie_film_eed08513-7f0e-4f02-9c7e-7c073ef45d57.png",
     [("eye and cheek", (570, 320)), ("lips", (790, 650)), ("hair in shadow", (1120, 840))]),
    ("parade.jpg", "Parade", "General/_MG_9373.jpg",
     [("face", (850, 690)), ("cords on black cloth", (800, 1000)), ("pale background", (1250, 200))]),
    ("flags.jpg", "Flags", "General/_MG_0330.jpg",
     [("stars on dark blue", (520, 380)), ("yellow and black", (590, 860)), ("sky", (1350, 230))]),
    ("warehouse.jpg", "Warehouse at dusk", "13 inch/3.jpg",
     [("brick and sign", (740, 600)), ("blue door in shadow", (150, 1100)), ("evening sky", (1110, 260))]),
    ("signpost.jpg", "Signpost in snow", "13 inch/3 2.jpg",
     [("lettering", (370, 280)), ("green sign", (650, 920)), ("trees and snow", (280, 1380))]),
    ("grey-ramp.png", "Grey ramp", "General/VerlaufGrau.png",
     [("dark end", (150, 100)), ("middle", (800, 290)), ("light steps", (1420, 1060))]),
]

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
PANEL_JSON = json.loads((ROOT / "spectra6_dither" / "panel-model.json").read_text())
INKS_PHOTO = np.array([PANEL_JSON["inks_as_photographed"][r] for r in ROLES], np.float32)

# ---------------------------------------------------------------------------------------------------- images


def to8(lin):
    return Image.fromarray((linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8))


def half(lin):
    """2x2 panel pixels -> 1 view pixel, averaged in linear light."""
    H, W = lin.shape[:2]
    ch = [Image.fromarray(np.ascontiguousarray(lin[..., c], np.float32)).resize((W // 2, H // 2), Image.BOX)
          for c in range(3)]
    return to8(np.stack([np.asarray(c) for c in ch], -1))


def stacked(top, bottom):
    """Stacked A/B layout: the middle half of the photo twice, one version above the other, divided by a 2 px white
    line, so both halves show the same content and refresh together."""
    H = top.shape[0]
    h = H // 2
    y0 = round(0.5 * (H - h))
    comp = np.concatenate([top[y0 : y0 + h], bottom[y0 : y0 + h]], 0)
    comp[h - 1 : h + 1] = 1  # white
    return comp


def crop_box(centre, shape):
    H, W = shape
    cw, chh = CROP if W >= H else CROP[::-1]
    x0 = int(np.clip(centre[0] - cw // 2, 0, W - cw))
    y0 = int(np.clip(centre[1] - chh // 2, 0, H - chh))
    return x0, y0, cw, chh


def build_image(file, title, upstream, crops):
    name = Path(file).stem
    ref8 = np.asarray(resize_cover(Image.open(ROOT / "samples" / file)))
    idx = {"ours": dither(ref8)}
    idx["epdoptimize"], opts = epdoptimize_app(ref8)
    # which version is on top of the A/B image: fixed per image, but not always the same one
    ours_top = hashlib.sha256(name.encode()).digest()[0] % 2 == 0
    order = ("ours", "epdoptimize") if ours_top else ("epdoptimize", "ours")
    device_image(stacked(idx[order[0]], idx[order[1]])).save(IMG / f"{name}.ab.device.png", optimize=True)

    ref_lin = srgb_to_linear(ref8 / 255.0)
    views = {"original": ref_lin}
    scores = {"model": {}, "flat": {}}
    for v in ("epdoptimize", "ours"):
        views[v] = render(idx[v])
        scores["model"][v] = round(score(views[v], ref_lin), 2)
        scores["flat"][v] = round(score(INKS_PHOTO[idx[v]], ref_lin), 2)
    boxes = [crop_box(c, ref8.shape[:2]) for _, c in crops]
    for v, lin in views.items():
        half(lin).save(IMG / f"{name}.{v}.webp", quality=90, method=6)
        for r, (x0, y0, cw, chh) in enumerate(boxes, 1):
            if v == "original":
                Image.fromarray(ref8[y0 : y0 + chh, x0 : x0 + cw]).save(IMG / f"{name}.{r}.{v}.png", optimize=True)
            else:
                to8(lin[y0 : y0 + chh, x0 : x0 + cw]).save(IMG / f"{name}.{r}.{v}.png", optimize=True)
    print(f"{name:10s} model: epdoptimize {scores['model']['epdoptimize']:5.2f} ours {scores['model']['ours']:5.2f}  "
          f"flat: epdoptimize {scores['flat']['epdoptimize']:5.2f} ours {scores['flat']['ours']:5.2f}  "
          f"{opts['imageKind']} {opts['options'].get('errorDiffusionMatrix')}", flush=True)
    return {"name": name, "file": file, "title": title, "upstream": upstream, "w": ref8.shape[1], "h": ref8.shape[0],
            "scores": scores, "epdoptimize": opts, "ab": order,
            "crops": [{"label": lab, "box": b} for (lab, _), b in zip(crops, boxes)]}


# ---------------------------------------------------------------------------------------------------- page
COLS = [("original", "Photo"), ("epdoptimize", "epdoptimize 1.3.0, app settings"), ("ours", "spectra6-dither")]
KERNELS = {"floydSteinberg": "Floyd-Steinberg", "stucki": "Stucki"}


def section(im, lazy):
    n, W, H = im["name"], im["w"], im["h"]
    ld = ' loading="lazy"' if lazy else ""
    e = html.escape
    kern = KERNELS.get(im["epdoptimize"]["options"].get("errorDiffusionMatrix"), "?")
    cw, chh = im["crops"][0]["box"][2:]
    boxes = "".join(
        f'<span class="box{" on" if r == 1 else ""}" data-r="{r}" style="left:{100 * x / W:.3f}%;'
        f'top:{100 * y / H:.3f}%;width:{100 * w / W:.3f}%;height:{100 * h / H:.3f}%"></span>'
        for r, (x, y, w, h) in enumerate((c["box"] for c in im["crops"]), 1))
    fars, crops = [], []
    for key, label in COLS:
        if key == "original":
            sub = f'{W}&times;{H}, <a href="../samples/{e(im["file"])}">file</a>'
        else:
            sub = f'&Delta;E<sub>00</sub> {im["scores"]["model"][key]:.2f}'
        kernel = f' <span class="m">({kern})</span>' if key == "epdoptimize" else ""
        fars.append(f'<figure class="c {key}"><div class="view"><img src="img/{n}.{key}.webp"{ld} width="{W // 2}" '
                    f'height="{H // 2}" alt="{e(label)}: {e(im["title"])}">{boxes}</div>'
                    f'<figcaption>{e(label)}{kernel}<br><span class="m">{sub}</span></figcaption></figure>')
        crops.append(f'<div class="c {key}"><div class="crop"><img src="img/{n}.1.{key}.png"{ld} data-col="{key}" '
                     f'width="{cw}" height="{chh}" style="--w:{cw}" alt="{e(label)}, 1:1 detail"></div></div>')
    buttons = "".join(f'<button type="button" data-r="{r}" aria-pressed="{str(r == 1).lower()}">'
                      f'{e(c["label"])}</button>' for r, c in enumerate(im["crops"], 1))
    names = dict(COLS)
    top, bottom = im["ab"]
    return f"""
<section id="{n}" data-name="{n}" data-boxes='{json.dumps([c["box"] for c in im["crops"]])}'>
  <h2>{e(im["title"])} <span class="m">sample-images/{e(im["upstream"])}</span></h2>
  <div class="row">{"".join(fars)}</div>
  <div class="regions"><span class="m">1:1 detail:</span> {buttons}</div>
  <div class="row">{"".join(crops)}</div>
  <div class="ab m"><a href="img/{n}.ab.device.png">Stacked A/B device PNG</a> (the middle half of the photo
  twice, one version above the other) <details><summary>which half is which</summary>top: {names[top]},
  bottom: {names[bottom]}</details></div>
</section>"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spectra6-dither comparison</title>
<style>
:root { --bg: #fafaf8; --fg: #1c1c1e; --m: #62626a; --line: #dcdcd6; --hi: #d9480f; --z: 2; color-scheme: light dark; }
@media (prefers-color-scheme: dark) { :root { --bg: #161618; --fg: #e6e6e8; --m: #9a9aa2; --line: #36363c; --hi: #ff8a4c; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; line-height: 1.3; margin: 8px 0 12px; }
h2 { font-size: 18px; margin: 0 0 10px; }
h2 .m { overflow-wrap: anywhere; }
a { color: inherit; }
.m { color: var(--m); font-size: 13px; font-weight: normal; }
.intro { max-width: 72ch; }
.note { max-width: 72ch; border-left: 3px solid var(--hi); padding: 2px 0 2px 12px; }
.zoom { position: sticky; top: 0; z-index: 2; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line); }
.zoom span { white-space: nowrap; }
section { padding: 20px 0 8px; border-bottom: 1px solid var(--line); }
.row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items: start; }
figure { margin: 0; }
figcaption { margin-top: 4px; font-size: 14px; }
.view { position: relative; cursor: crosshair; line-height: 0; }
.view img { width: 100%; height: auto; display: block; }
.box { position: absolute; outline: 1px dashed rgba(255,255,255,.8); box-shadow: 0 0 0 1px rgba(0,0,0,.5); pointer-events: none; }
.box.on { outline: 2px solid var(--hi); }
.regions { margin: 14px 0 8px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
button { font: inherit; font-size: 13px; color: var(--fg); background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 3px 9px; cursor: pointer; }
button[aria-pressed=true] { border-color: var(--hi); color: var(--hi); }
.crop { overflow: auto; max-height: 75vh; line-height: 0; }
.crop img { width: calc(var(--w) * var(--z) * 1px); max-width: none; height: auto; image-rendering: pixelated; }
.ab { margin: 10px 0 0; }
details { display: inline-block; margin-left: 8px; }
summary { cursor: pointer; }
@media (max-width: 700px) {
  .row { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .row > .original { grid-column: 1 / -1; }
}
</style>
</head>
<body>
<main>
<h1>OpenPaper L: epdoptimize 1.3.0 as the paperlesspaper app runs it, and spectra6-dither</h1>
<p class="intro">paperlesspaper's sample images at the panel's size (1600&times;1200, 150 ppi): the photo, epdoptimize
1.3.0 with the settings the paperlesspaper app uses (the kernel its suggestion picked is in brackets), and
spectra6-dither. The first row shows each whole image at half size, every pixel the linear-light average of 2&times;2
panel pixels; click it to choose the 1:1 detail below. Panel white is shown as screen white. &Delta;E<sub>00</sub> is
the mean S&#8209;CIELAB CIEDE2000 difference from the photo at 50 cm, lower is closer. Each image links a stacked A/B
device PNG for a real frame; <code>python -m spectra6_dither photo.jpg out.png</code> makes a full-frame one.</p>
<p class="note">Both dithered images are simulated with the same panel model, measured on one OpenPaper L: a lighter pixel
loses part of its area to darker neighbours, yellow spreads into white, green in fine patterns reads yellow-green.
Both show the model's black, which is purple-blue because the glare removed from the measured inks is assumed
neutral. epdoptimize's own preview paints every pixel in a flat palette colour and cannot show any of this. Our
dithering is designed for this model, so the pictures and these scores lean its way; the README has a score without
the model and the real-frame A/B results.</p>
<div class="zoom">Detail zoom <span class="m">(screen px per panel px)</span> <span><button type="button" data-z="1">1&times;</button>
<button type="button" data-z="2">2&times;</button> <button type="button" data-z="3">3&times;</button></span></div>
__SECTIONS__
<p class="m">Built by scripts/build_compare.py. Sample images by paperlesspaper (see samples/NOTICE), not covered by
this repository's license.</p>
</main>
<script>
const root = document.documentElement;
function setZoom(z) {
  root.style.setProperty("--z", z);
  document.querySelectorAll("[data-z]").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.z === z)));
}
setZoom(innerWidth >= 900 ? 2 : 1);
document.querySelectorAll("[data-z]").forEach(b => b.addEventListener("click", () => setZoom(+b.dataset.z)));

document.querySelectorAll("section[data-name]").forEach(sec => {
  const name = sec.dataset.name, boxes = JSON.parse(sec.dataset.boxes);
  const crops = [...sec.querySelectorAll(".crop")];
  function pick(r) {
    sec.querySelectorAll(".crop img").forEach(img => {
      img.src = `img/${name}.${r}.${img.dataset.col}.png`;
    });
    sec.querySelectorAll(".box").forEach(b => b.classList.toggle("on", +b.dataset.r === r));
    sec.querySelectorAll(".regions button").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.r === r)));
  }
  sec.querySelectorAll(".regions button").forEach(b => b.addEventListener("click", () => pick(+b.dataset.r)));
  sec.querySelectorAll(".view").forEach(v => v.addEventListener("click", ev => {
    const rc = v.getBoundingClientRect(), img = v.querySelector("img");
    const x = (ev.clientX - rc.left) / rc.width * img.naturalWidth * 2;
    const y = (ev.clientY - rc.top) / rc.height * img.naturalHeight * 2;
    let best = 1, bd = Infinity;
    boxes.forEach(([x0, y0, w, h], i) => {
      const d = (x0 + w / 2 - x) ** 2 + (y0 + h / 2 - y) ** 2;
      if (d < bd) { bd = d; best = i + 1; }
    });
    pick(best);
  }));
  // the three crops scroll together
  let busy = false;
  crops.forEach(c => c.addEventListener("scroll", () => {
    if (busy) return;
    busy = true;
    crops.forEach(o => { if (o !== c) { o.scrollLeft = c.scrollLeft; o.scrollTop = c.scrollTop; } });
    requestAnimationFrame(() => { busy = false; });
  }));
});
</script>
</body>
</html>
"""


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    images = [build_image(*sample) for sample in SAMPLES]
    mean = {k: {v: round(float(np.mean([im["scores"][k][v] for im in images])), 2) for v in ("epdoptimize", "ours")}
            for k in ("model", "flat")}
    (OUT / "scores.json").write_text(json.dumps(
        {"mean": mean, **{im["name"]: {**im["scores"], "epdoptimize_options": im["epdoptimize"]} for im in images}},
        indent=1) + "\n")
    page = PAGE.replace("__SECTIONS__", "".join(section(im, k > 0) for k, im in enumerate(images)))
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
