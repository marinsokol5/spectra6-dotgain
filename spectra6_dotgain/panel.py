"""The measured panel model: what the OpenPaper L shows for a map of inks, dot darkening included.

A pixel of ink p loses part of its area to its 4-neighbours n that spread into it. Each neighbour contributes
D[p, n] (0 = no spread, about 1 = black next to white), k is the sum over the 4 neighbours, and the lost area is
L(k), linear between the measured values at k = 0..4. The lost area takes the colours of the spreading neighbours,
in proportion to their D. Everything is adapted linear RGB with panel white == 1.

The numbers are in panel-model.json; README.md says how they were measured.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROLES = ["black", "white", "blue", "green", "red", "yellow"]
# what the frame expects for each ink, in ROLES order
DEVICE = np.array([[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]], np.uint8)
MODEL = Path(__file__).with_name("panel-model.json")
N4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
KS = np.arange(5.0)


def load(path=MODEL):
    """{"P": 6x3 ink colours, "Lk": L(k) at k = 0..4, "D": 6x6 spread weights (row: pixel ink, column: neighbour)}."""
    m = json.loads(Path(path).read_text())
    return {
        "P": np.array([m["inks"][r] for r in ROLES], float),
        "Lk": np.concatenate([[0.0], np.asarray(m["L"], float)]),
        "D": np.asarray(m["D"], float),
    }


def render(idx, m=None):
    """Index map (HxW, ROLES indices) -> adapted linear RGB the panel shows (HxWx3 float32, white == 1)."""
    m = m or load()
    P, Lk = m["P"], m["Lk"]
    out = P[idx].astype(np.float32)
    pad = np.pad(idx, 1, mode="edge")
    H, W = idx.shape
    nbs = [pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W] for dy, dx in N4]
    D = m["D"].astype(np.float32)
    ds = [D[idx, nb] for nb in nbs]
    k = np.sum(ds, 0)
    loss = np.interp(k, KS, Lk).astype(np.float32)
    share = np.where(k > 0, loss / np.maximum(k, 1e-6), 0).astype(np.float32)
    out *= (1 - loss)[..., None]
    Pf = P.astype(np.float32)
    for d, nb in zip(ds, nbs):
        out += (share * d)[..., None] * Pf[nb]
    return out


def device_image(idx):
    """Index map in content orientation -> the 1600x1200 RGB image the frame takes.
    Portrait content is rotated 90 degrees clockwise into the landscape panel."""
    im = Image.fromarray(DEVICE[idx])
    if im.height > im.width:
        im = im.transpose(Image.Transpose.ROTATE_270)
    assert im.size == (1600, 1200), im.size
    return im


def indices_from_device(rgb8):
    """Device-colour image -> index map (nearest device colour)."""
    d = ((rgb8[..., None, :].astype(np.float32) - DEVICE.astype(np.float32)[None, None]) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8)
