# NOTE: adapted from Michael's pr

import cv2
import numpy as np
from pathlib import Path
from typing import Literal, TypedDict

from rapid.helpers import get_zcam_bandset
from marslab.imgops.imgutils import crop, eightbit
from asdf.zcam_bandset import ZcamBandSet
import asdf_settings.metadata
from asdf_settings import rapidlooks

from .utils import get_rgb_stretch

ZCAM_CROP = rapidlooks.CROP_SETTINGS["crop"]
SHARED_BANDS = {"L": "L1", "R": "R1"}
BAD_FLAGS = ('bad', 'no_signal', 'hot')
BAD_PIXMAP_VALUES = tuple(
    i + 1 for i, f in enumerate(asdf_settings.metadata.PIXEL_FLAG_NAMES)
    if f in BAD_FLAGS
)

class LoadResult(TypedDict):
    cube: np.ndarray
    base_bands: dict[str, np.ndarray]
    bandset: ZcamBandSet
    homography_tmask: np.ndarray
    rgb_img: np.ndarray

def load_cube(iof_path, seq_id, obs_ix, do_apply_pixmaps, ignore_bayers):
    
    cube = []
    
    # load left and right data cubes
    search_path = Path(iof_path)
    
    bs = get_zcam_bandset(search_path, seq_id=seq_id, observation_ix=obs_ix, load=False)
    filts = bs.metadata["BAND"].sort_values()
    if ignore_bayers is True:
        filts = filts.loc[~filts.str.contains("0")].reset_index()
    bs.load("all")
    bs.bulk_debayer("all")
    base_bands = {b: crop(bs.get_band(b), ZCAM_CROP).copy() for b in filts}
    if do_apply_pixmaps is True:
        pixmaps = {
            b: crop(bs.pixmaps[b], ZCAM_CROP).copy()
            for b in sorted(bs.metadata["FILTER"].unique())
        }
        # NOTE: applying NaN values here is ugly, but cv2.warpPerspective will
        # not respect a MaskedArray's mask and warping a boolean array is
        # questionable
        bands = apply_pixmaps(base_bands, pixmaps)
    else:
        pixmaps = None
        bands = base_bands
    l_cube = np.array([a for b, a in bands.items() if b.startswith("L")])
    r_cube = np.array([a for b, a in bands.items() if b.startswith("R")])
    
    # store rgb image of scene (used for segmentation)
    rgb_img = get_rgb_stretch(r_cube)
    
    # b/c we might have applied NaN values to bands of the cube, always use
    # the original bands for computing homography (NaNs make cv2's SIFT
    # implementation unhappy)
    h_matrix = compute_homography(
        base_bands[SHARED_BANDS["L"]], base_bands[SHARED_BANDS["R"]],
    )
    l_cube_mapped = apply_homography(l_cube, h_matrix, r_cube[0].shape)
    
    # get index of last shared band between left/right cameras
    last_shared_band_index = (
        sorted(bs.raw)
        .index(SHARED_BANDS['L'])
    )
    
    # mask the overlap bewteen the cameras
    homography_tmask = np.array(l_cube_mapped[last_shared_band_index] == 0)
    
    # average bands shared between left/right cameras (Bayer + 800nm)
    for band in range(last_shared_band_index + 1):
        band_avg = (l_cube_mapped[band] + r_cube[band]) / 2
        cube.append(band_avg)

    # store left bands
    l_num_bands = l_cube.shape[0]
    for band in range(last_shared_band_index + 1, l_num_bands):
        cube.append(l_cube_mapped[band])

    # store right bands
    r_num_bands = r_cube.shape[0]
    for band in range(last_shared_band_index + 1, r_num_bands):
        cube.append(r_cube[band])

    return {
        'cube': np.array(cube),
        'base_bands': base_bands,
        'bandset': bs,
        'homography_tmask': homography_tmask,
        'rgb_img': rgb_img
    }

# NOTE: This approach is not robust to parallax...
def apply_homography(
    src_cube: np.ndarray, hmat: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    # NOTE: in this pipeline, shape _should_ always be the same as
    # src_cube.shape
    cube_transformed = []
    for band in range(src_cube.shape[0]):
        spec_slice = src_cube[band]
        warped_img = cv2.warpPerspective(spec_slice, hmat, (shape[1], shape[0]))
        cube_transformed.append(warped_img)
    return np.array(cube_transformed)


def compute_homography(
    src: np.ndarray, dst: np.ndarray, prestretch: int = 1
) -> np.ndarray:
    """
    Compute a homography matrix that maps src to dst. src and dst must be
    2D ndarrays.
    """
    src, dst = map(lambda a: eightbit(a, prestretch), (src, dst))

    # detect features and compute descriptors
    sift = cv2.SIFT_create()
    src_keypoints, src_descriptors = sift.detectAndCompute(src, None)
    dst_keypoints, dst_descriptors = sift.detectAndCompute(dst, None)
    # match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(src_descriptors, dst_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance
    # extract matched keypoints
    src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    # compute homography matrix
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]


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
