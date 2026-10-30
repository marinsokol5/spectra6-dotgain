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

