"""Dithering for the paperlesspaper OpenPaper L (E Ink Spectra 6, 1600x1200) that plans for measured dot darkening."""
import numpy as np

from .mbed import FS, mbed_core
from .panel import DEVICE, ROLES, device_image, indices_from_device, load, render
from .prep import PANEL, linear_to_srgb, resize_cover
from .target import make_target

__all__ = ["dither", "device_image", "indices_from_device", "load", "render", "resize_cover", "linear_to_srgb",
           "DEVICE", "PANEL", "ROLES"]


def dither(ref8, m=None):
    """ref8: HxWx3 uint8 sRGB at panel size (see resize_cover) -> HxW index map into ROLES."""
    m = m or load()
    t = make_target(ref8, m["P"].astype(np.float32))
    return mbed_core(t, m["P"], m["Lk"], m["D"], FS)
