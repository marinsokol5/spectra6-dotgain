"""Photo -> panel-size sRGB image, and the colour conversions the rest of the package uses."""
import io

import numpy as np
from PIL import Image, ImageCms, ImageOps

PANEL = (1600, 1200)  # OpenPaper L, landscape; portrait content is 1200x1600 until it is rotated for the device


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


# OKLab (Ottosson), input linear sRGB
M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
M1i = np.linalg.inv(M1)
M2i = np.linalg.inv(M2)


def lin_to_oklab(lin):
    lms = lin @ M1.T
    lms_ = np.cbrt(lms)
    return (lms_ @ M2.T).astype(np.float32)


def oklab_to_lin(lab):
    lms_ = lab @ M2i.T
    lms = lms_**3
    return (lms @ M1i.T).astype(np.float32)


_SRGB = ImageCms.createProfile("sRGB")


def to_srgb(im):
    """Honour an embedded ICC profile (e.g. Display P3) by converting to sRGB, and the EXIF orientation."""
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            im = ImageCms.profileToProfile(im.convert("RGB"), src, _SRGB,
                                           renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")
        except Exception:  # a broken profile: treat the pixels as sRGB
            pass
    return im.convert("RGB")


def resize_cover(im, size=PANEL):
    """Cover-crop to the panel's aspect, then Lanczos resize in linear light. Portrait photos stay portrait."""
    im = to_srgb(im)
    W, H = size
    if im.height > im.width:
        W, H = H, W
    scale = max(W / im.width, H / im.height)
    cw, ch = round(W / scale), round(H / scale)
    left = (im.width - cw) // 2
    top = (im.height - ch) // 2
    im = im.crop((left, top, left + cw, top + ch))
    lin = srgb_to_linear(np.asarray(im, dtype=np.float32) / 255.0)
    chans = [np.asarray(Image.fromarray(lin[..., c], mode="F").resize((W, H), Image.LANCZOS)) for c in range(3)]
    out = linear_to_srgb(np.clip(np.stack(chans, -1), 0, 1))
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))
