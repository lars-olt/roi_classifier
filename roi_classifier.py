"""
A module for classifying ROI.
"""
from pathlib import Path
from typing import TypedDict

from asdf.zcam_bandset import ZcamBandSet
from asdf_settings import rapidlooks
import cv2
from marslab.imgops.imgutils import crop, eightbit
import matplotlib.pyplot as plt
import numpy as np
from rapid.helpers import get_zcam_bandset
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, center_of_mass, find_objects, label
from sklearn.cluster import KMeans

from mask_helpers import apply_pixmaps

SHARED_BANDS = {"L": "L1", "R": "R1"}
WLS = [630, 544, 480, 800, 754, 677, 605, 528, 442, 866, 910, 939, 978, 1022]
ZCAM_CROP = rapidlooks.CROP_SETTINGS["crop"]


class LoadResult(TypedDict):
    l_cube: np.ndarray
    r_cube: np.ndarray
    bands: dict[str, np.ndarray]
    base_bands: dict[str, np.ndarray]
    bandset: ZcamBandSet
    pixmaps: dict[str, np.ndarray] | None


def load_data(
    iof_path: str | Path,
    seq_id: str | int | None = None,
    obs_ix: int | None = None,
    do_apply_pixmaps: bool = True,
    ignore_bayers: bool = False
):
    bs = get_zcam_bandset(iof_path, seq_id=seq_id, observation_ix=obs_ix,
                          load=False)
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
    l_cube = np.array([a for b, a in bands.items() if b.startswith('L')])
    r_cube = np.array([a for b, a in bands.items() if b.startswith('R')])
    return {
        "l_cube": l_cube,
        "r_cube": r_cube,
        "bands": bands,
        "base_bands": base_bands,
        "pixmaps": pixmaps,
        "bandset": bs
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
        warped_img = cv2.warpPerspective(spec_slice, hmat,
                                         (shape[1], shape[0]))
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
    src_pts = np.float32(
        [src_keypoints[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32(
        [dst_keypoints[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2
    )
    # compute homography matrix
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]


def mask_cube(cube, mask):
    """applies mask to the cube."""
    
    stacked_mask = np.repeat(mask[np.newaxis, :], cube.shape[0], axis=0)
    masked_cube = np.ma.masked_array(cube, mask=stacked_mask)

    return masked_cube


def compress_cube(masked_cube):
    """create an array with only vlaid pixels.
    returns new compressed array and valid pixel locations."""
    
    # gets valid spatial locations shared by all bands
    # corresponds to valid pixels
    spatial_mask = ~masked_cube.mask.any(
        axis=0
    )

    # extract valid pixels per band
    compressed_cube = masked_cube[
        :, spatial_mask
    ].data  # data grouped by bands, removing masked values

    # extract valid spatial indices
    pixel_locations = np.argwhere(spatial_mask)

    return compressed_cube, pixel_locations


def uncompress(compressed_data, pixel_locations, shape):
    """remaps values in compressed_data to masked array with shape."""
    
    reconstructed = np.ma.masked_all(shape, dtype=compressed_data.dtype)
    is_cube = len(shape) == 3

    if is_cube:
        bands, _, _ = shape
        pixel_indices = tuple(pixel_locations.T)
        for band in range(bands):
            reconstructed[band][pixel_indices] = compressed_data[band]
    else:
        pixel_indices = tuple(pixel_locations)
        reconstructed[pixel_indices] = compressed_data

    return reconstructed


def apply_kmeans_to_masked(masked_array, k, seed=42):
    """applies k-means algorithm to masked array."""
    
    # compress array to contain only unmasked values
    spatial_mask = ~masked_array.mask.any(
        axis=0
    )
    valid_pixels = masked_array[:, spatial_mask].data  # get valid pixels per band
    compressed_cube = valid_pixels.T.astype(np.float32)  # reshape to (pixels, bands)

    # apply kmeans
    k_means = KMeans(
        n_clusters=k, random_state=seed
    )  # NOTE: random state set to make deterministic
    classifications = k_means.fit_predict(compressed_cube)

    # uncompress k-means classifications to orriginal masked shape
    _, h, w = masked_array.shape
    pixel_indices = np.argwhere(spatial_mask).T
    uncompressed_classifications = uncompress(classifications, pixel_indices, (h, w))

    return uncompressed_classifications


def filter_connected_components(clustered, min_area, max_area):
    """filters clustered regions based on their contiguous area.
        returns mask of regions with areas within [mix, max]"""
    cube_data = clustered.data
    cube_mask = clustered.mask

    mask = np.zeros(cube_data.shape)

    for label_value in np.unique(cube_data):
        
        # finds connected features in array
        binary_mask = (cube_data == label_value) & ~cube_mask
        labeled_mask, _ = label(binary_mask)
        features = find_objects(labeled_mask)
        
        temp_mask = np.zeros_like(binary_mask)
        
        # populate temp mask
        for i, feature in enumerate(features, start=1):
            # extract the component using the slice
            component = labeled_mask[feature] == i
            component_size = np.sum(component)

            # retain the component if it satisfies the size constraints
            if min_area <= component_size <= max_area:
                temp_mask[feature][component] = 1

        # store temp mask
        mask = np.logical_or(mask, temp_mask)
    
    filled_mask = binary_fill_holes(mask)

    return filled_mask


def largest_rect_around_center(mask, center):
    """finds the largest rectangle at the given center location that fits within the mask"""
    row, col = int(center[0]), int(center[1])
    total_rows = mask.shape[0]
    total_cols = mask.shape[1]

    # initialize boundaries to point
    left, right, top, bottom = col, col, row, row
    
    left_inbounds = True
    right_inbounds = True
    top_inbounds = True
    bottom_inbounds = True
    
    # expand in each direction until image edge or mask edge is reached
    # TODO: this is not robust to weird regions
    while (left_inbounds or right_inbounds or top_inbounds or bottom_inbounds):
        left_inbounds = (left > 0) and np.all(mask[top : bottom + 1, left - 1 : right + 1] == 1)
        right_inbounds = (right < total_cols - 1) and np.all(mask[top : bottom + 1, left : right + 2] == 1)
        top_inbounds = (top > 0) and np.all(mask[top - 1 : bottom + 1, left : right + 1] == 1)
        bottom_inbounds = (bottom < total_rows - 1) and np.all(mask[top : bottom + 2, left : right + 1] == 1)
        
        if left_inbounds:
            left -= 1
        if right_inbounds:
            right += 1
        if top_inbounds:
            top -= 1
        if bottom_inbounds:
            bottom += 1

    return (left, top, right, bottom)


def find_band_roi(binary_array, center_proximity=100, density_threshold=0.7):
    """identifies regions of interest from k-means classification mask."""

    h, w = binary_array.shape
    img_center = (int(h / 2), int(w / 2))

    # compute density map based on distance from edges
    distance_transform = ndimage.distance_transform_edt(binary_array)

    # normalize distances (ensure greater than 0)
    if distance_transform.max() > 0:
        normalized_distance = distance_transform / distance_transform.max()
    else:
        normalized_distance = np.zeros_like(distance_transform)

    # set density to zero outside the mask
    density_within_mask = (
        normalized_distance * binary_array
    )  # multiply by the mask to retain values only within it

    hotspots = density_within_mask > density_threshold

    labeled_hotspots, _ = label(hotspots)

    # Compute centers of mass for each label
    centers = center_of_mass(
        labeled_hotspots > 0,
        labels=labeled_hotspots,
        index=range(1, labeled_hotspots.max() + 1),
    )

    # group centers together that are close to eachother
    # TODO: make this efficent. there are duplicated calculations here... 
    to_remove = []
    for i, i_center in enumerate(centers):
        if i in to_remove:
            continue
        for j, j_center in enumerate(centers):
            if i == j:
                continue

            y_diff = (i_center[0] - j_center[0]) ** 2
            x_diff = (i_center[1] - j_center[1]) ** 2

            diff = (y_diff + x_diff) ** 0.5

            if diff < center_proximity:
                to_remove.append(j)

    centers_to_remove = [centers[i] for i in to_remove]
    groupped_centers = [c for c in centers if c not in centers_to_remove]

    # place a rectangle for each center of mass
    band_rects = []
    for center in groupped_centers:

        # calculate distance from center
        y_diff = (center[0] - img_center[0]) ** 2
        x_diff = (center[1] - img_center[1]) ** 2
        center_dist = (y_diff + x_diff) ** 0.5

        # find the largest rectangle centered at this point
        left, top, right, bottom = largest_rect_around_center(binary_array, center)

        # compute width and height of the rectangle
        width = right - left + 1
        height = bottom - top + 1
        area = width * height

        band_rects.append((area, int(center_dist), (left, top, width, height)))

    # return (band_rects, groupped_centers, density_within_mask)  # for testing...
    return band_rects


def get_dist_between(pt1, pt2):
    """computes distance between two points."""
    
    a = (pt1[0] - pt2[0]) ** 2
    b = (pt1[1] - pt2[1]) ** 2
    dist_between = (a + b) ** 0.5
    
    return int(dist_between)


def add_rois(
    cluster,
    min_region_sz,
    max_region_sz,
    center_thresh_dif,
    density_threshold,
    edge_prox=80,
):
    """classifies valid rois based on passed params"""
    rois = []

    data = cluster.data
    y_dim, x_dim = cluster.shape

    for centroid in np.unique(data):

        # compute possible roi regions
        band = (data == centroid) & ~cluster.mask  # mask data for particular classification
        img = np.array(
            binary_fill_holes(band), dtype=np.uint8
        )  # Fill holes in the mask
        rects = find_band_roi(
            img[1:], center_proximity=100, density_threshold=density_threshold
        )

        # filter roi selections
        min_center_dist = -1
        min_center_loc = ()
        roi_coords = []

        for area, center_dist, coords in rects:
            # filter by region size
            if (area < min_region_sz) or (area > max_region_sz):
                continue

            center_x = coords[0] + int(coords[2] / 2)
            center_y = coords[1] + int(coords[3] / 2)

            # filter by edge proximity
            if (
                (center_x - edge_prox) < 0
                or (center_x + edge_prox) > x_dim
                or (center_y - edge_prox) < 0
                or (center_y + edge_prox) > y_dim
            ):
                continue

            # *if* multiple regions for a band exist, take closest to image center
            if min_center_dist == -1 or center_dist < min_center_dist:

                if len(roi_coords) > 0:
                    # compare distance between previously stored roi and current roi
                    dist_between = get_dist_between(
                        min_center_loc, (center_x, center_y)
                    )
                    if dist_between > center_thresh_dif:
                        roi_coords = [coords] + roi_coords
                    else:
                        # replace previously minimized coord
                        roi_coords[0] = (
                            coords  # first element cooresponds to the minimized roi
                        )
                else:
                    # first coord
                    roi_coords.append(coords)

                min_center_dist = center_dist
                min_center_loc = (center_x, center_y)

        # add back ROIs that are greater than minimum dist from selected ROI
        # TODO: there is a better way of doing this.
        for area, center_dist, coords in rects:
            # filter by region size
            if (area < min_region_sz) or (area > max_region_sz):
                continue

            center_x = coords[0] + int(coords[2] / 2)
            center_y = coords[1] + int(coords[3] / 2)

            # filter by edge proximity
            if (
                (center_x - edge_prox) < 0
                or (center_x + edge_prox) > x_dim
                or (center_y - edge_prox) < 0
                or (center_y + edge_prox) > y_dim
            ):
                continue

            if center_dist > min_center_dist and len(min_center_loc) == 2:
                # check if distance is greater than threshold
                dist_between = get_dist_between(min_center_loc, (center_x, center_y))
                if dist_between > center_thresh_dif and (coords not in roi_coords):
                    roi_coords.append(coords)

        rois += roi_coords

    return rois


def average_spectra(data, rectangles):
    """
    calculate the average spectra for each rectangle in the hyperspectral cube.
    """
    averaged_spectra = []
    std_spectra = []

    for x1, y1, x2, y2 in rectangles:
        # Extract the region within the rectangle
        region = data[:, y1 : y2 + 1, x1 : x2 + 1]

        # Average over the spatial dimensions (height, width)
        avg_spectrum = region.mean(axis=(1, 2))
        averaged_spectra.append(avg_spectrum)

        std_spectrum = region.std(axis=(1, 2))
        std_spectra.append(std_spectrum)

    return np.ma.getdata(averaged_spectra), np.ma.getdata(std_spectra)


def plot_spectra(spectra, stds, colors, markers):
    """
    plot the averaged spectra for each rectangle.
    """
    plt.figure(figsize=(7, 7))

    bayer_sorted_indices = np.argsort(WLS[:3])
    non_bayer_sorted_indices = np.argsort(WLS[3:]) + 3

    color_i = 0
    marker_i = 0
    for i, spectrum in enumerate(spectra):

        # cycles colors if need be
        if color_i == len(colors):
            color_i = 0
            
        # cycles markers if need be
        if marker_i == len(markers):
            marker_i = 0

        curr_color = colors[color_i]

        # plot non-bayer bands
        nb_wls = np.array(WLS)[non_bayer_sorted_indices]
        nb_data = spectrum[non_bayer_sorted_indices]
        plt.errorbar(
            nb_wls,
            nb_data,
            yerr=stds[i][non_bayer_sorted_indices],
            fmt="-",
            ecolor=curr_color,
            capsize=3,
            color=curr_color,
            marker=markers[marker_i]
        )

        # plot bayer bands
        b_wls = np.array(WLS)[bayer_sorted_indices]
        b_data = spectrum[bayer_sorted_indices]
        plt.plot(b_wls, b_data, "+", color=curr_color)  # ,

        color_i += 1
        marker_i += 1

    plt.xlabel("wavelength (nm)")
    plt.ylabel("R* = IOF/cos(θ)")
    plt.ylim(top=0.5)  # makes y-axis max (0.5)
    plt.show()
