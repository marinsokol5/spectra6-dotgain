"""python -m spectra6_dotgain photo.jpg out.png [--sim preview.png]"""
import argparse

import numpy as np
from PIL import Image

from . import device_image, dither, linear_to_srgb, render, resize_cover


def main():
    ap = argparse.ArgumentParser(prog="python -m spectra6_dotgain",
                                 description="Photo -> device PNG for the OpenPaper L (1600x1200, 6 device colours).")
    ap.add_argument("photo")
    ap.add_argument("out", help="device PNG; portrait photos are rotated 90 degrees clockwise into 1600x1200")
    ap.add_argument("--sim", help="also write what the panel model predicts the frame shows (photo orientation)")
    a = ap.parse_args()
    idx = dither(np.asarray(resize_cover(Image.open(a.photo))))
    device_image(idx).save(a.out, optimize=True)
    if a.sim:
        Image.fromarray((linear_to_srgb(render(idx)) * 255 + 0.5).astype(np.uint8)).save(a.sim)


if __name__ == "__main__":
    main()
