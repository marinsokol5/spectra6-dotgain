"""Photo -> continuous target colour per pixel, inside what the panel's inks can mix.

All in OKLab of the white-adapted panel (source white == panel white):

  1. tone: a local tone map. The tone curve acts on an edge-preserving base layer (self-guided filter, radius 100 px)
     and the detail on top is added back unchanged, so shadows are lifted without flattening local contrast.
     The curve is a 50/50 blend of black-point compensation (squeeze [0, 1] into [L_black, L_white]) and a power toe
     that keeps L above L_black + 0.10 and rolls the rest smoothly onto panel black.
  2. gamut: each pixel moves along a ray from an anchor on the panel's black-white axis until it is inside the
     convex hull of the inks in linear light (everything dithering can mix). The anchor's lightness is the pixel's
     own (colours go greyer, not lighter) unless the hue's most saturated panel colour is darker, then 20 % of the
     way towards it.
  3. inky: strongly saturated source colours (more than 55 % of the most saturated sRGB colour of their hue, fully
     at 85 %) are pulled towards the most saturated colour the panel can show at that hue, so a red sign stays red
     instead of turning dusty pink. Skin, most skies and foliage are below the threshold and untouched.

The result is adapted linear RGB (panel white == 1).
"""
import math

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter
from scipy.spatial import ConvexHull

from .prep import M1i, M2i, lin_to_oklab, oklab_to_lin, srgb_to_linear

BPC = 0.5  # half black-point compensation: with none, original L* 0-20 all landed on panel black on the frame
KNEE = 0.10  # the toe maps source L 0 .. L_black + KNEE onto L_black .. L_black + KNEE, and leaves L above alone
ALPHA_DOWN = 0.2  # move towards the hue's cusp when it is darker than the pixel (deep blue sky)
ALPHA_UP = 0.0  # never lighten a colour to gain chroma
GUIDED_R, GUIDED_EPS = 100, 0.002
INKY_S0, INKY_S1 = 0.55, 0.85


class Hull:
    """Convex hull of the inks in adapted linear RGB, with its OKLab black-white axis and max chroma per (L, hue)."""

    def __init__(self, inks):
        self.pts = np.asarray(inks, dtype=np.float64)
        self.eq = ConvexHull(self.pts).equations  # n.x + d <= 0 inside
        ok = lin_to_oklab(self.pts.astype(np.float32))
        self.white_ok = ok[ok[:, 0].argmax()].astype(np.float64)
        self.black_ok = ok[ok[:, 0].argmin()].astype(np.float64)
        self._lut()

    def inside(self, lin, tol=1e-6):
        return (lin @ self.eq[:, :3].T + self.eq[:, 3] <= tol).all(-1)

    def neutral_ab(self, L):
        """a, b of the panel's own neutral axis (the line black -> white) at OKLab lightness L."""
        t = np.clip((L - self.black_ok[0]) / (self.white_ok[0] - self.black_ok[0]), 0, 1)
        return self.black_ok[1:] + t[..., None] * (self.white_ok[1:] - self.black_ok[1:])

    def _lut(self, nL=96, nH=180):
        Ls = np.linspace(self.black_ok[0], self.white_ok[0], nL)
        hs = np.linspace(-np.pi, np.pi, nH, endpoint=False)
        LL, HH = np.meshgrid(Ls, hs, indexing="ij")
        lo = np.zeros_like(LL)
        hi = np.full_like(LL, 0.5)
        n_ab = self.neutral_ab(LL)
        for _ in range(28):
            mid = (lo + hi) / 2
            ok = np.stack([LL, n_ab[..., 0] + mid * np.cos(HH), n_ab[..., 1] + mid * np.sin(HH)], -1)
            ins = self.inside(oklab_to_lin(ok.astype(np.float32)).astype(np.float64))
            lo = np.where(ins, mid, lo)
            hi = np.where(ins, hi, mid)
        self.lut_L = Ls
        self.lut_h = hs
        self.lut_C = lo


def project_into_hull(lin, hull):
    """Pull any pixel still outside the hull towards the hull centroid until it is inside."""
    c = hull.pts.mean(0)
    out = lin.astype(np.float64).copy()
    flat = out.reshape(-1, 3)
    vals = flat @ hull.eq[:, :3].T + hull.eq[:, 3]
    bad = (vals > 1e-7).any(-1)
    if bad.any():
        p = flat[bad]
        d = p - c
        num = -(c @ hull.eq[:, :3].T + hull.eq[:, 3])  # > 0
        den = d @ hull.eq[:, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(den > 0, num[None, :] / den, np.inf).min(-1)
        t = np.clip(t, 0, 1) * (1 - 1e-6)
        flat[bad] = c + t[:, None] * d
    return out.reshape(lin.shape).astype(np.float32)


def guided_self(I, r, eps):
    """Self-guided filter (He et al.) with box radius r: edge-preserving base layer."""
    size = 2 * int(r) + 1
    m = uniform_filter(I, size, mode="reflect")
    m2 = uniform_filter(I * I, size, mode="reflect")
    var = m2 - m * m
    a = var / (var + eps)
    b = m - a * m
    return uniform_filter(a, size, mode="reflect") * I + uniform_filter(b, size, mode="reflect")


def tone_curve(L, Lb, Lw):
    Lbpc = Lb + L * (Lw - Lb)
    top = Lb + KNEE
    x = np.clip(L / top, 0, 1)
    soft = Lb + KNEE * x ** (top / KNEE)  # C1 at top: value top, slope 1
    Lnob = np.where(L >= top, L, soft)
    Lp = BPC * Lbpc + (1 - BPC) * Lnob
    return np.clip(Lp, Lb, Lw)


_M1i = M1i.copy()  # numba freezes globals at compile time
_M2i = M2i.copy()


@njit(cache=True)
def _ok2lin(L, a, b, out):
    l_ = _M2i[0, 0] * L + _M2i[0, 1] * a + _M2i[0, 2] * b
    m_ = _M2i[1, 0] * L + _M2i[1, 1] * a + _M2i[1, 2] * b
    s_ = _M2i[2, 0] * L + _M2i[2, 1] * a + _M2i[2, 2] * b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    out[0] = _M1i[0, 0] * l + _M1i[0, 1] * m + _M1i[0, 2] * s
    out[1] = _M1i[1, 0] * l + _M1i[1, 1] * m + _M1i[1, 2] * s
    out[2] = _M1i[2, 0] * l + _M1i[2, 1] * m + _M1i[2, 2] * s


@njit(cache=True)
def _inside(p, eq, tol):
    for f in range(eq.shape[0]):
        if eq[f, 0] * p[0] + eq[f, 1] * p[1] + eq[f, 2] * p[2] + eq[f, 3] > tol:
            return False
    return True


@njit(cache=True)
def gamut_map(Lp, A_, B_, alpha_dn, alpha_up, eq, black_ok, white_ok, cuspL, iters, centroid):
    """Lp: tone-mapped OKLab L (HxW); A_, B_: source OKLab a, b. Returns adapted linear RGB inside the hull."""
    H, W = Lp.shape
    nH = cuspL.shape[0]
    out = np.empty((H, W, 3), np.float32)
    Lb = black_ok[0]
    Lw = white_ok[0]
    q = np.empty(3)
    for y in range(H):
        for x in range(W):
            L = min(max(Lp[y, x], Lb), Lw)
            a = A_[y, x]
            b = B_[y, x]
            h = math.atan2(b, a)
            hf = (h + math.pi) / (2 * math.pi) * nH
            fl = math.floor(hf)
            j0 = int(fl) % nH
            j1 = (j0 + 1) % nH
            f = hf - fl
            cl = cuspL[j0] * (1 - f) + cuspL[j1] * f
            al = alpha_dn if cl < L else alpha_up
            # anchor on the neutral axis
            L0 = (1 - al) * L + al * cl
            L0 = min(max(L0, Lb + 1e-3), Lw - 1e-3)
            t = (L0 - Lb) / (Lw - Lb)
            t = min(max(t, 0.0), 1.0)
            Aa = black_ok[1] + t * (white_ok[1] - black_ok[1])
            Ab = black_ok[2] + t * (white_ok[2] - black_ok[2])
            DL = L - L0
            Da = a - Aa
            Db = b - Ab
            # the pixel is at s = 1 on the ray anchor + s * D: find the hull boundary (doubling, then bisection)
            lo = 0.0
            hi = 1.0
            while hi < 64.0:
                _ok2lin(L0 + hi * DL, Aa + hi * Da, Ab + hi * Db, q)
                if _inside(q, eq, 1e-7):
                    lo = hi
                    hi *= 2.0
                else:
                    break
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                _ok2lin(L0 + mid * DL, Aa + mid * Da, Ab + mid * Db, q)
                if _inside(q, eq, 1e-7):
                    lo = mid
                else:
                    hi = mid
            sb = max(lo, 1e-6)
            s = min(1.0 / sb, 1.0) * sb  # hard clip at the boundary
            _ok2lin(L0 + s * DL, Aa + s * Da, Ab + s * Db, q)
            # safety: pull towards the hull centroid if bisection left it a hair outside
            tmin = 1.0
            bad = False
            for fct in range(eq.shape[0]):
                v = eq[fct, 0] * q[0] + eq[fct, 1] * q[1] + eq[fct, 2] * q[2] + eq[fct, 3]
                if v > 1e-7:
                    bad = True
            if bad:
                dx0 = q[0] - centroid[0]
                dx1 = q[1] - centroid[1]
                dx2 = q[2] - centroid[2]
                for fct in range(eq.shape[0]):
                    num = -(eq[fct, 0] * centroid[0] + eq[fct, 1] * centroid[1] + eq[fct, 2] * centroid[2] + eq[fct, 3])
                    den = eq[fct, 0] * dx0 + eq[fct, 1] * dx1 + eq[fct, 2] * dx2
                    if den > 0:
                        tt = num / den
                        if tt < tmin:
                            tmin = tt
                tmin = max(tmin, 0.0) * (1 - 1e-6)
                q[0] = centroid[0] + tmin * dx0
                q[1] = centroid[1] + tmin * dx1
                q[2] = centroid[2] + tmin * dx2
            out[y, x, 0] = q[0]
            out[y, x, 1] = q[1]
            out[y, x, 2] = q[2]
    return out


NH = 360
_srgb_cmax = None


def srgb_cmax_lut():
    """Max OKLab chroma of sRGB per hue bin (from the saturated faces of the RGB cube)."""
    global _srgb_cmax
    if _srgb_cmax is None:
        t = np.linspace(0, 1, 257)
        g = np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
        pts = []
        for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            c = np.zeros((len(g), 3))
            c[:, perm[0]] = 1.0  # one channel full, one free, one empty: the most saturated colours
            c[:, perm[1]] = g[:, 0]
            c[:, perm[2]] = 0.0
            pts.append(c)
            c2 = c.copy()
            c2 *= g[:, 1:2]  # and their darker versions
            pts.append(c2)
        rgb = np.concatenate(pts)
        ok = lin_to_oklab(srgb_to_linear(rgb.astype(np.float32)))
        C = np.hypot(ok[:, 1], ok[:, 2])
        h = np.arctan2(ok[:, 2], ok[:, 1])
        j = (((h + np.pi) / (2 * np.pi)) * NH).astype(int) % NH
        lut = np.zeros(NH)
        np.maximum.at(lut, j, C)
        for k in np.where(lut == 0)[0]:
            lut[k] = max(lut[(k - 1) % NH], lut[(k + 1) % NH])
        _srgb_cmax = lut
    return _srgb_cmax


def cusp_lut(hull):
    """Linear-RGB cusp (most saturated panel colour) per hue bin of the hull LUT."""
    j = np.argmax(hull.lut_C, axis=0)
    Lc = hull.lut_L[j]
    Cc = hull.lut_C[j, np.arange(len(hull.lut_h))]
    hs = hull.lut_h
    n_ab = hull.neutral_ab(Lc)
    ok = np.stack([Lc, n_ab[:, 0] + Cc * np.cos(hs), n_ab[:, 1] + Cc * np.sin(hs)], -1)
    lin = project_into_hull(oklab_to_lin(ok.astype(np.float32))[None], hull)[0].astype(np.float64)
    return hs, lin


def inky_blend(t, src_ok, hull):
    """Blend t towards the hue's cusp, by how saturated the source is relative to sRGB's most saturated colour."""
    C = np.hypot(src_ok[..., 1], src_ok[..., 2])
    h = np.arctan2(src_ok[..., 2], src_ok[..., 1])
    sl = srgb_cmax_lut()
    hb = ((h + np.pi) / (2 * np.pi) * NH).astype(int) % NH
    rel = C / np.maximum(sl[hb], 1e-6)
    x = np.clip((rel - INKY_S0) / (INKY_S1 - INKY_S0), 0, 1)
    w = x * x * (3 - 2 * x)
    hs, cusp = cusp_lut(hull)
    nH = len(hs)
    hf = (h + np.pi) / (2 * np.pi) * nH
    j0 = np.floor(hf).astype(int) % nH
    j1 = (j0 + 1) % nH
    f = (hf - np.floor(hf))[..., None]
    cp = cusp[j0] * (1 - f) + cusp[j1] * f
    return (1 - w[..., None]) * t + w[..., None] * cp


def make_target(ref8, inks):
    """ref8: HxWx3 uint8 sRGB. inks: 6x3 adapted linear RGB (float32, white == 1). Returns HxWx3 float64."""
    hull = Hull(inks)
    Lb, Lw = float(hull.black_ok[0]), float(hull.white_ok[0])
    lin = srgb_to_linear(ref8 / 255.0)
    ok = lin_to_oklab(np.clip(lin, 0, 1)).astype(np.float64)
    L = np.clip(ok[..., 0], 0, 1)
    base = guided_self(L, GUIDED_R, GUIDED_EPS)
    Lp = np.clip(tone_curve(base, Lb, Lw) + (L - base), Lb, Lw)
    cusp = hull.lut_L[np.argmax(hull.lut_C, axis=0)].astype(np.float64)
    t = gamut_map(np.ascontiguousarray(Lp), np.ascontiguousarray(ok[..., 1]), np.ascontiguousarray(ok[..., 2]),
                  ALPHA_DOWN, ALPHA_UP, hull.eq.astype(np.float64), hull.black_ok, hull.white_ok, cusp, 16,
                  hull.pts.mean(0))
    return np.ascontiguousarray(inky_blend(t.astype(np.float64), ok, hull), np.float64)
