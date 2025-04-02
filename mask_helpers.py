from __future__ import annotations

from typing import Literal

import asdf_settings.meta
import numpy as np

BAD_FLAGS = ('bad', 'no_signal', 'hot')
BAD_PIXMAP_VALUES = tuple(
    i + 1 for i, f in enumerate(asdf_settings.meta.PIXEL_FLAG_NAMES)
    if f in BAD_FLAGS
)


def make_eye_mask(
    pixmaps: dict[str, np.ndarray], eye: Literal["L", "R"],
) -> np.ndarray:
    pixmaps = {k: v for k, v in pixmaps.items() if k.startswith(eye)}
    pixmaps = [np.isin(v, BAD_PIXMAP_VALUES) for v in pixmaps.values()]
    return np.any(np.dstack(pixmaps), axis=2)


def apply_pixmaps(
    bands: dict[str, np.ndarray], pixmaps: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    l_pix_mask = make_eye_mask(pixmaps, "L")
    r_pix_mask = make_eye_mask(pixmaps, "R")
    outbands = {}
    for k, v in bands.items():
        mask = l_pix_mask if k.startswith("L") else r_pix_mask
        outbands[k] = np.where(mask, np.nan, bands[k])
    return outbands
