# spectra6-dither

We dither photos for a paperlesspaper OpenPaper L (13.3" E Ink Spectra 6, 1600x1200 at 150 ppi) against a model of
how its dots really render, measured on our frame. This repository holds that code, the measurements behind it, and
a page that shows our output next to epdoptimize 1.3.0 as the paperlesspaper app runs it. It is what we use, not a
request to change epdoptimize's default.

Comparison page: https://marinsokol5.github.io/spectra6-dither/compare/

![Chart patches photographed on the panel](docs/checkerboard-vs-blocks.png)

Two 50 % black chart patches photographed on the panel (the camera does not resolve the 1-px checkerboard). The
strips show each patch's mean over 6 photos (luminance, white = 1); per-pixel mixing predicts 0.59 for both.

## What the panel does

We photographed a calibration chart on our frame in daylight (6 phone RAW photos, decoded linearly; glare lifts
black to luminance 0.17). The measuring code is not included.

- A lighter pixel loses part of its area to darker 4-neighbours: 20, 27, 33 and 51 % with 1, 2, 3 and 4 darker
  neighbours (rms error 0.018 in luminance).
- At 50 % black the 1-px checkerboard reads 0.37; 1-px horizontal lines (0.50), vertical lines (0.51) and 4x4
  blocks (0.49) agree within the spread between photos. Per-pixel mixing predicts 0.58 to 0.59 for all four.
- Fitted on the chart's 12 two-ink checkerboards, a white pixel loses area to a yellow neighbour with weight 0.90
  and to a black one with 1.07; a rule based on luminance difference gives yellow 0.10.
- Green in fine patterns reads yellower than the solid patch: b* +7.0 for a green/white checkerboard, +0.8
  predicted. On a photo of the frame our A/B band read 8.8 b* yellower than the app's, twice what the models
  predicted; with green made yellow-green (Lab about 63/-24/24) the model gives 7.7. Dithering for that model cut
  the yellow by 3.8 b* on the next photo (predicted 3.7 to 4.8).

Mean CIEDE2000 between `spectra6_dither/panel.py` and the photos, with the inks as photographed:

| model | 17 mixed patches (fitted on these) | 81 ramp segments (not fitted) |
|---|---|---|
| per-pixel mixing | 6.47 | 4.47 |
| darkening, spread by luminance difference | 4.58 | 2.58 |
| darkening, spread fitted per ink pair | 4.01 | 2.45 |

The shipped `panel-model.json` is the last row plus two unfitted changes: green as above, and a uniform glare
subtracted so black reads 0.06 (our guess at a typical room). That glare is assumed neutral, so removing it moves
black from Lab 48.4/5.0/-8.1 to 29.4/10.9/-16.2 and shadows look purple in the simulations. epdoptimize's
flat-colour preview cannot show any of this.

## What ours does

`spectra6_dither/` is the code we run, taken from our lab code without changing the maths: about 6 s per image on
an M2 (7.6 s on the first run, while numba compiles).

1. Cover-crop to 4:3, Lanczos resize in linear light (`prep.py`).
2. Target (`target.py`, OKLab): a tone curve (half black-point compensation, half a power toe) on an
   edge-preserving base layer, detail kept; colours moved into what the inks can mix, keeping lightness unless the
   hue's most saturated panel colour is darker (then 20 % towards it); sources above 55 % of sRGB's maximum chroma
   pulled towards that colour, fully at 85 %.
3. Dither (`mbed.py`): serpentine Floyd-Steinberg in linear light, choosing each ink by what the panel model says
   this pixel and its placed neighbours will show (closest in OKLab) and diffusing the error relative to that, not
   to the flat ink colour.

## Results

Ours is designed for the model the page simulates, so the pictures and scores lean our way. Mean S-CIELAB
CIEDE2000 against the photo at 50 cm is 12.4 for ours and 19.3 for epdoptimize. With per-pixel mixing of the inks as
photographed, no darkening at all, ours is still closer: 20.6 against 25.8 (per image in `compare/scores.json`).
Part of the gap is tone: epdoptimize can tone-map, but the app doesn't use it.

The independent evidence is the real frame. We ran 8 blind A/B tests (placement random, key sealed until judged) on
5 of our own photos, judged by eye by the frame's owner, sometimes with family. A is an early version with its own
tone and gamut mapping but no panel model; B is the predecessor of this code, with the panel model and half
black-point compensation; C is this code.

- B beat the app 3 of 3. A first model-based version without black-point compensation also beat it, but crushed
  the shadows the app kept (original L* 0-10 and 10-20 both landed on panel black, L* 29.5; half compensation puts
  them at 33.9 and 38.1). The app's version had blown highlights and pale colours.
- B beat A 2 of 2. A beat the app once and lost twice, once in a side-by-side layout showing different parts of
  the photo.
- C beat B in its only test; it has not been tested on the frame against the app.

## Running it

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # tested with Python 3.12
.venv/bin/python -m spectra6_dither photo.jpg device.png --sim preview.png
(cd scripts && npm ci) && .venv/bin/python scripts/build_compare.py
```

`device.png` is a full-frame image in the 6 device colours (portrait photos rotated 90° clockwise) that
paperlesspaper's API accepts as `pictureDevice`. With the pinned versions on an M2 Mac, `build_compare.py`
reproduces every file in `compare/` byte for byte.

## epdoptimize as the app runs it

`scripts/epdoptimize_app.mjs` (epdoptimize 1.3.0, pinned in `scripts/package-lock.json`) follows paperlesspaper-web's
image editor:

1. The editor works from a copy of the photo capped at 1280 px on the long edge, which Fabric scales back up with
   image smoothing; we resize our 1600x1200 image to 1280 and back with Pillow's bilinear filter.
2. `suggestCanvasProcessingOptions(canvas, aitjcizeSpectra6Palette, {intent: "natural"})`; the app keeps only its
   dither keys. It analyses a 1280 px copy, we the full canvas; both give the same keys here.
3. `ditherImage` with the JS engine (the app's default `auto` runs the WASM kernel; on these images the output is
   byte-identical): serpentine error diffusion with the suggested kernel (Floyd-Steinberg; Stucki for the parade
   photo) on 8-bit gamma-encoded sRGB, clamped at every step, nearest colour by RGB distance to black #020202, white
   #BEC8C8, blue #05409E, green #27663C, red #871300, yellow #CDCA00.
4. `replaceColors` maps those to the device colours.

The app does not use the tone mapping and dynamic range compression the suggestion proposes, so neither does this
comparison.

## What we don't know

- One panel, 6 RAW photos of one chart in one room; the glare we subtract is an assumption.
- The green change rests on one photo of the frame, checked on one more; the solid green patch reads teal.
- The A/B results are 8 tests on our own photos, judged by the frame's owner. Faces were judged only there: the
  sample photos have no colour skin (the page's portrait is AI-generated).
- The model sees 4-neighbours and one darkening curve; viewing angle, temperature and ghosting are unmeasured.

## License

Code: Apache-2.0 (`LICENSE`). The sample images are paperlesspaper's, not covered by it (`samples/NOTICE`).
epdoptimize is a dependency, not copied; the palette values come from its `default-palettes.json`.
