// epdoptimize 1.3.0 the way the paperlesspaper phone app runs it (paperlesspaper-web, image editor generatePreview):
//   suggestCanvasProcessingOptions(canvas, aitjcizeSpectra6Palette, {intent: "natural"})
//   -> keep only the dither keys of the suggestion (the app's pickDitherOptions drops the preset, tone mapping,
//      dynamic range compression and levels)
//   -> ditherImage(..., {processingEngine: "js", adjustmentEngine: "js", ...those keys, palette})
//      (the app's default, useAcceleratedPreviewProcessing, passes "auto", which runs epdoptimize's WASM kernel;
//      on the sample images both engines give byte-identical output)
//   -> replaceColors: the palette's display colours -> the 6 device colours.
// The input PNG is already at panel size and resampled like the app's editor (see build_compare.py).
// A minimal canvas shim stands in for the browser canvas (no node-canvas needed).
//
// usage: node epdoptimize_app.mjs in.png device.png      (prints the dither options used, as JSON)
import fs from "node:fs";
import { PNG } from "pngjs";
import * as epd from "epdoptimize";

const [, , inPath, devicePath] = process.argv;
const APP_KEYS = ["ditheringType", "errorDiffusionMatrix", "algorithm", "serpentine", "orderedDitheringType",
  "orderedDitheringMatrix", "randomDitheringType", "colorMatching", "sampleColorsFromImage", "numberOfSampleColors"];
const pick = (o) => Object.fromEntries(Object.entries(o ?? {}).filter(([k]) => APP_KEYS.includes(k)));

class ShimImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}
globalThis.ImageData ??= ShimImageData;

const makeCanvas = (width, height, data) => {
  const canvas = {
    width,
    height,
    _data: data ?? new Uint8ClampedArray(width * height * 4),
    getContext() {
      return {
        canvas,
        getImageData(x, y, w, h) {
          if (x === 0 && y === 0 && w === canvas.width && h === canvas.height) {
            return new ShimImageData(new Uint8ClampedArray(canvas._data), w, h);
          }
          const out = new Uint8ClampedArray(w * h * 4);
          for (let row = 0; row < h; row++) {
            const src = ((y + row) * canvas.width + x) * 4;
            out.set(canvas._data.subarray(src, src + w * 4), row * w * 4);
          }
          return new ShimImageData(out, w, h);
        },
        putImageData(img) {
          if (img.width !== canvas.width || img.height !== canvas.height) {
            canvas.width = img.width;
            canvas.height = img.height;
          }
          canvas._data = new Uint8ClampedArray(img.data);
        },
        createImageData(w, h) {
          return new ShimImageData(new Uint8ClampedArray(w * h * 4), w, h);
        },
        drawImage() {
          throw new Error("drawImage not supported by shim");
        },
        fillRect() {},
        set fillStyle(_) {},
      };
    },
  };
  return canvas;
};

const png = PNG.sync.read(fs.readFileSync(inPath));
const src = makeCanvas(png.width, png.height, new Uint8ClampedArray(png.data));
const palette = epd.aitjcizeSpectra6Palette;
const suggestion = epd.suggestCanvasProcessingOptions(src, palette, { intent: "natural" });
const options = { processingEngine: "js", adjustmentEngine: "js", ...pick(suggestion.ditherOptions), palette };
const dithered = makeCanvas(png.width, png.height);
await epd.ditherImage(src, dithered, options);
const device = makeCanvas(png.width, png.height);
epd.replaceColors(dithered, device, palette);

const out = new PNG({ width: device.width, height: device.height });
out.data = Buffer.from(device._data.buffer, device._data.byteOffset, device._data.byteLength);
fs.writeFileSync(devicePath, PNG.sync.write(out));
const { palette: _p, ...printable } = options;
console.log(JSON.stringify({ imageKind: suggestion.imageKind ?? suggestion.classification?.kind, options: printable }));
