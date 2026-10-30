"""Model-based error diffusion: Floyd-Steinberg against what the panel will show, not against flat ink colours.

On the OpenPaper L a lighter pixel loses part of its area to darker 4-neighbours (panel.py), so fine patterns come
out darker than plain error diffusion plans for. Here, for every candidate ink c of pixel p:

  * the rendered colours of p (given its already placed neighbours) and of the placed neighbours whose
    rendering changes (the pixel above, now final because all 4 of its neighbours are known, and the previous pixel
    in the scan, which gets one more neighbour) are recomputed with the panel model;
  * every change caused by the choice is charged to p, so each edge effect is accounted for exactly once;
  * the ink is chosen that minimises the OKLab distance between what p should contribute and what it does,
    and p's error is diffused as usual (serpentine scan).
"""
import numpy as np
from numba import njit

# Floyd-Steinberg: (dx, dy, weight)
FS = np.array([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], dtype=np.float64)
FS[:, 2] /= FS[:, 2].sum()
ERR_CLIP = 0.5  # the buffered target may run this far outside [0, 1] per channel


@njit(cache=True)
def _cbrt(x):
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)


@njit(cache=True)
def _oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = _cbrt(l), _cbrt(m), _cbrt(s)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


@njit(cache=True)
def _render_px(idx, y, x, P, Lk, D, out):
    """Rendered colour of placed pixel (y, x) given its placed 4-neighbours (idx -1 = not placed yet)."""
    H, W = idx.shape
    c = idx[y, x]
    k = 0.0
    g0 = 0.0
    g1 = 0.0
    g2 = 0.0
    for t in range(4):
        ny = y + (0, 0, 1, -1)[t]
        nx = x + (1, -1, 0, 0)[t]
        if ny < 0 or ny >= H or nx < 0 or nx >= W:
            continue
        n = idx[ny, nx]
        if n < 0:
            continue
        d = D[c, n]
        if d > 0:
            k += d
            g0 += d * P[n, 0]
            g1 += d * P[n, 1]
            g2 += d * P[n, 2]
    if k <= 0:
        out[0], out[1], out[2] = P[c, 0], P[c, 1], P[c, 2]
        return
    kk = min(k, 4.0)
    i = min(int(kk), 3)
    f = kk - i
    loss = Lk[i] * (1 - f) + Lk[i + 1] * f
    s = loss / k
    out[0] = P[c, 0] * (1 - loss) + s * g0
    out[1] = P[c, 1] * (1 - loss) + s * g1
    out[2] = P[c, 2] * (1 - loss) + s * g2


@njit(cache=True)
def mbed_core(target, P, Lk, D, kern):
    H, W, _ = target.shape
    K = P.shape[0]
    buf = target.astype(np.float64).copy()
    idx = -np.ones((H, W), np.int16)
    rend = np.zeros((H, W, 3))
    nk = kern.shape[0]
    rp = np.zeros(3)
    rq = np.zeros(3)
    rh = np.zeros(3)
    best_rp = np.zeros(3)
    best_dq = np.zeros(3)
    best_dh = np.zeros(3)
    for y in range(H):
        rev = y % 2 == 1
        for i in range(W):
            x = W - 1 - i if rev else i
            hx = x + 1 if rev else x - 1  # previous pixel in this row's scan
            has_h = 0 <= hx < W
            has_q = y > 0
            want0 = min(max(buf[y, x, 0], -ERR_CLIP), 1 + ERR_CLIP)
            want1 = min(max(buf[y, x, 1], -ERR_CLIP), 1 + ERR_CLIP)
            want2 = min(max(buf[y, x, 2], -ERR_CLIP), 1 + ERR_CLIP)
            bd = 1e30
            best = 0
            for c in range(K):
                idx[y, x] = c
                _render_px(idx, y, x, P, Lk, D, rp)
                dq0 = dq1 = dq2 = 0.0
                dh0 = dh1 = dh2 = 0.0
                if has_q:
                    _render_px(idx, y - 1, x, P, Lk, D, rq)
                    dq0, dq1, dq2 = rq[0] - rend[y - 1, x, 0], rq[1] - rend[y - 1, x, 1], rq[2] - rend[y - 1, x, 2]
                if has_h:
                    _render_px(idx, y, hx, P, Lk, D, rh)
                    dh0, dh1, dh2 = rh[0] - rend[y, hx, 0], rh[1] - rend[y, hx, 1], rh[2] - rend[y, hx, 2]
                # what p should show once the neighbours' changes are accounted for, vs what it shows
                a0 = max(want0 - dq0 - dh0, 0.0)
                a1 = max(want1 - dq1 - dh1, 0.0)
                a2 = max(want2 - dq2 - dh2, 0.0)
                L1, A1, B1 = _oklab(a0, a1, a2)
                L2, A2, B2 = _oklab(max(rp[0], 0.0), max(rp[1], 0.0), max(rp[2], 0.0))
                d = (L1 - L2) ** 2 + ((A1 - A2) ** 2 + (B1 - B2) ** 2)
                if d < bd:
                    bd = d
                    best = c
                    best_rp[0], best_rp[1], best_rp[2] = rp[0], rp[1], rp[2]
                    best_dq[0], best_dq[1], best_dq[2] = dq0, dq1, dq2
                    best_dh[0], best_dh[1], best_dh[2] = dh0, dh1, dh2
            idx[y, x] = best
            rend[y, x, 0], rend[y, x, 1], rend[y, x, 2] = best_rp[0], best_rp[1], best_rp[2]
            if has_q:
                for ch in range(3):
                    rend[y - 1, x, ch] += best_dq[ch]
            if has_h:
                for ch in range(3):
                    rend[y, hx, ch] += best_dh[ch]
            e0 = want0 - best_rp[0] - best_dq[0] - best_dh[0]
            e1 = want1 - best_rp[1] - best_dq[1] - best_dh[1]
            e2 = want2 - best_rp[2] - best_dq[2] - best_dh[2]
            for j in range(nk):
                dx = int(kern[j, 0])
                if rev:
                    dx = -dx
                nx = x + dx
                ny = y + int(kern[j, 1])
                if nx < 0 or nx >= W or ny >= H:
                    continue
                f = kern[j, 2]
                buf[ny, nx, 0] += e0 * f
                buf[ny, nx, 1] += e1 * f
                buf[ny, nx, 2] += e2 * f
    return idx.astype(np.uint8)
