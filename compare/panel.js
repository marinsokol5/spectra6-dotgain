// The panel model of spectra6_dotgain/panel.py, for the browser (and Node, for the test against Python).
// Same float32 arithmetic as numpy, in the same order, so a device image renders to the same 8-bit sRGB here as in
// Python. Everything is adapted linear RGB, panel white == 1.

export const ROLES = ["black", "white", "blue", "green", "red", "yellow"];
export const DEVICE = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]];
const f32 = Math.fround;
const N4 = [[0, 1], [0, -1], [1, 0], [-1, 0]]; // (dy, dx), in panel.py's order

export function loadModel(json) {
  const P = new Float32Array(18), D = new Float32Array(36);
  ROLES.forEach((r, i) => json.inks[r].forEach((v, c) => { P[i * 3 + c] = v; }));
  json.D.forEach((row, i) => row.forEach((v, j) => { D[i * 6 + j] = v; }));
  return { P, D, Lk: [0, ...json.L] };
}

// np.interp(k, [0, 1, 2, 3, 4], Lk) for one value, as numpy computes it (in double)
function interp(x, Lk) {
  if (x >= 4) return Lk[4];
  const j = Math.floor(x);
  if (x === j) return Lk[j];
  return (Lk[j + 1] - Lk[j]) * (x - j) + Lk[j];
}

// Index map (W*H, ROLES indices) -> Float32Array(W*H*3) of what the panel shows; edge pixels see themselves
// beyond the border, like np.pad(mode="edge").
export function render(idx, W, H, m) {
  const { P, D, Lk } = m;
  const out = new Float32Array(W * H * 3);
  const d = new Float32Array(4), nb = new Uint8Array(4), tiny = f32(1e-6);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x, p = idx[i];
      let k = 0;
      for (let t = 0; t < 4; t++) {
        const yy = Math.min(Math.max(y + N4[t][0], 0), H - 1), xx = Math.min(Math.max(x + N4[t][1], 0), W - 1);
        nb[t] = idx[yy * W + xx];
        d[t] = D[p * 6 + nb[t]];
        k = t === 0 ? d[0] : f32(k + d[t]);
      }
      const loss = f32(interp(k, Lk));
      const share = k > 0 ? f32(loss / Math.max(k, tiny)) : 0;
      const keep = f32(1 - loss);
      for (let c = 0; c < 3; c++) {
        let v = f32(P[p * 3 + c] * keep);
        for (let t = 0; t < 4; t++) v = f32(v + f32(f32(share * d[t]) * P[nb[t] * 3 + c]));
        out[i * 3 + c] = v;
      }
    }
  }
  return out;
}

// linear -> 8-bit sRGB, as (linear_to_srgb(v) * 255 + 0.5).astype(np.uint8) in prep.py
const T = f32(0.0031308), A = f32(12.92), E = f32(1 / 2.4), S = f32(1.055), O = f32(0.055);
export function srgb8(v) {
  const c = Math.min(Math.max(v, 0), 1);
  const s = c <= T ? f32(c * A) : f32(f32(S * f32(Math.pow(c, E))) - O);
  return Math.trunc(f32(f32(s * 255) + 0.5));
}

// 8-bit sRGB -> linear, as srgb_to_linear in prep.py
export const LINEAR = Float32Array.from({ length: 256 }, (_, i) => {
  const c = f32(i / 255);
  return c <= f32(0.04045) ? f32(c / f32(12.92)) : f32(Math.pow(f32(f32(c + O) / S), f32(2.4)));
});

// RGBA pixels of a device image -> ROLES indices (nearest device colour)
export function indices(rgba) {
  const n = rgba.length / 4, idx = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < 6; k++) {
      const dr = rgba[i * 4] - DEVICE[k][0], dg = rgba[i * 4 + 1] - DEVICE[k][1], db = rgba[i * 4 + 2] - DEVICE[k][2];
      const dd = dr * dr + dg * dg + db * db;
      if (dd < bd) { bd = dd; best = k; }
    }
    idx[i] = best;
  }
  return idx;
}

// The A/B image for a real frame: the middle half of the photo twice, one version above the other, divided by a 2 px
// white line, in the panel's 1600x1200 (portrait rotated 90 degrees clockwise). top, bottom: index maps, W x H.
export function abDevice(top, bottom, W, H) {
  const h = H >> 1, y0 = Math.round(0.5 * (H - h)), comp = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const sy = y0 + (y < h ? y : y - h);
    comp.set((y < h ? top : bottom).subarray(sy * W, (sy + 1) * W), y * W);
  }
  comp.fill(1, (h - 1) * W, (h + 1) * W);
  const dw = Math.max(W, H), dh = Math.min(W, H), rgb = new Uint8Array(dw * dh * 3);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      rgb.set(DEVICE[W === dw ? comp[y * W + x] : comp[(dw - 1 - x) * W + y]], (y * dw + x) * 3);
    }
  }
  return { rgb, w: dw, h: dh };
}

// RGB PNG (8 bit, no filter) for the A/B download
const CRC = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function chunk(type, data) {
  const b = new Uint8Array(12 + data.length), v = new DataView(b.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) b[4 + i] = type.charCodeAt(i);
  b.set(data, 8);
  let c = 0xffffffff;
  for (let i = 4; i < 8 + data.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
  v.setUint32(8 + data.length, (c ^ 0xffffffff) >>> 0);
  return b;
}
export async function pngRGB(rgb, w, h) {
  const raw = new Uint8Array((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) raw.set(rgb.subarray(y * w * 3, (y + 1) * w * 3), y * (w * 3 + 1) + 1);
  const z = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate")))
    .arrayBuffer());
  const ihdr = new Uint8Array(13), v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const parts = [sig, chunk("IHDR", ihdr), chunk("IDAT", z), chunk("IEND", new Uint8Array(0))];
  return new Blob(parts, { type: "image/png" });
}
