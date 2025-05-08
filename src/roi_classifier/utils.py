import numpy as np
from .constraints import RGB_MAPPING
from marslab.imgops.imgutils import enhance_color
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from kneed import KneeLocator

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
    spatial_mask = ~masked_cube.mask.any(axis=0)

    # extract valid pixels per band
    compressed_cube = masked_cube[
        :, spatial_mask
    ].data  # data grouped by bands, removing masked values

    # extract valid spatial indices
    pixel_locations = np.argwhere(spatial_mask)

    return compressed_cube, pixel_locations


def uncompress_cube(compressed_data, pixel_locations, shape):
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


def get_center_of_mass(masked_arr):

    # compute density map based on distance from edges (exact euclidean distance transform)
    distance_transform = distance_transform_edt(masked_arr)

    # normalize distances (ensure greater than 0)
    normalized_distance = distance_transform / distance_transform.max()

    # apply orriginal array as mask
    # (we are only interested in density within the target region)
    density_within_mask = normalized_distance * masked_arr

    # compute highest density location
    highest_density_loc = np.where(density_within_mask == 1)

    # return only one center of mass
    # TODO: more sophisticated selection...
    cxs, cys = highest_density_loc
    center_of_mass = (int(cxs[0]), int(cys[0]))

    return center_of_mass


def largest_rect_around_center(mask, center):
    """finds the largest rectangle at the given center location that fits within the mask"""
    row, col = int(center[0]), int(center[1])
    total_rows = mask.shape[0]
    total_cols = mask.shape[1]

    # initialize boundaries to point
    left = right = col
    top = bottom = row

    left_inbounds = True
    right_inbounds = True
    top_inbounds = True
    bottom_inbounds = True

    # expand in each direction until image edge or mask edge is reached
    # TODO: this is not robust to weird regions
    while left_inbounds or right_inbounds or top_inbounds or bottom_inbounds:
        left_inbounds = (left > 0) and np.all(
            mask[top : bottom + 1, left - 1 : right + 1] == 1
        )
        right_inbounds = (right < total_cols - 1) and np.all(
            mask[top : bottom + 1, left : right + 2] == 1
        )
        top_inbounds = (top > 0) and np.all(
            mask[top - 1 : bottom + 1, left : right + 1] == 1
        )
        bottom_inbounds = (bottom < total_rows - 1) and np.all(
            mask[top : bottom + 2, left : right + 1] == 1
        )

        if left_inbounds:
            left -= 1
        if right_inbounds:
            right += 1
        if top_inbounds:
            top -= 1
        if bottom_inbounds:
            bottom += 1

    return (left, top, right, bottom)


def get_rgb_stretch(cube):
    img = {}
    for i, color in enumerate(RGB_MAPPING):
        img[color] = cube[i]
    
    mapped_img = [img['R'], img['G'], img['B']]
    rgb = np.ma.masked_invalid(np.stack(mapped_img, axis=-1))
    rgb_stretch = enhance_color(rgb, bounds=(0, 1), stretch=0.1)
    
    return rgb_stretch


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

def auto_k_elbow(X, k_range=range(5, 20)):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    return kl.knee if kl.knee else k_range[np.argmin(np.gradient(inertias))]

def cluster_spectra(spectra):
    n_clusters = auto_k_elbow(spectra)
    k_means = KMeans(
        n_clusters=n_clusters, random_state=42
    )  # NOTE: random state set to make deterministic
    classifications = k_means.fit_predict(spectra)
    
    return classifications