import numpy as np

# NOTE: Neccesary on windows for sklearn.cluster.KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"

from .constraints import WLS, COLORS
from .utils import (
    compress_cube,
    get_center_of_mass,
    largest_rect_around_center,
    mask_cube,
    uncompress_cube
)
from scipy.ndimage import binary_opening
from sklearn.cluster import KMeans

def get_potential_rois(segmented_img, masked_cube, edge_offset, allowed_variance):
    
    # compress cube
    # original_cube_shape = masked_cube.shape
    # compressed_cube, pixel_mapping = compress_cube(masked_cube)
    
    full_tmask = masked_cube.mask[0]  # NOTE: same mask coppied over all bands
    
    # find set of all rois
    rois = []
    for region_i in range(len(segmented_img)):  # loop physical SAM regions

        # spectrally cluster
        region = [segmented_img == region_i]
        cluster_ret = cluster_region(region, full_tmask, masked_cube, edge_offset, allowed_variance)
        
        if cluster_ret == None:
            continue

        clusters, k = cluster_ret

        # identify roi for each cluster
        for cluster in range(k):

            slice = (clusters.data == cluster) & ~clusters.mask
            _, roi = get_roi(slice)

            rois.append(roi)
    
    # TODO: finish
    
    return np.array(rois)


def cluster_region(
    region_mask, full_mask, spectral_difference, edge_offset, allowed_variance
):

    k = 1
    variance = 0
    prev_classification = []
    k_found = False

    # remove any shadow regions
    cluster_mask = region_mask[0].copy()
    cluster_mask[full_mask] = 0

    # add offset to edges
    max_y, max_x = cluster_mask.shape
    cluster_mask[:, :edge_offset] = 0  # left edge
    cluster_mask[:, (max_x - edge_offset) :] = 0  # right edge
    cluster_mask[:edge_offset, :] = 0  # top edge
    cluster_mask[(max_y - edge_offset) :, :] = 0  # bottom edge

    # erosion/dilation to remove any hairline artifacts from segmentation
    erosion_kernel = (5, 5)
    cleaned_mask = binary_opening(cluster_mask, structure=np.ones(erosion_kernel))
    # cleaned_mask = cluster_mask  # TODO: just to test, delete

    area = np.count_nonzero(cleaned_mask)
    # print(f'area={area}')

    if area == 0:
        # print('Empty segment.')
        return None

    # keep full region for pebbles
    if area < 4000:
        # print('Found pebble.')
        pebble_mask = np.ma.masked_array(
            np.zeros_like(cleaned_mask).astype(np.int32), mask=~cleaned_mask
        )
        return pebble_mask, k

    # TODO: filter by mask area here...

    # get masked image
    masked_img = mask_cube(spectral_difference, ~cluster_mask)

    # step k value until variance is above ALLOWED_VARIANCE
    # (previous classification is returned)
    while not k_found:

        # classify regions
        curr_classification = apply_kmeans_to_masked(masked_img, k)

        # compute variance of regions
        # (low variance -> more homogeneous)
        variance = np.var(curr_classification)
        # print(f'k={k}, var={variance}')

        # check termination
        k_found = variance >= allowed_variance
        if not k_found:
            prev_classification = curr_classification
            k += 1
        else:
            k -= 1

    # print(f'found k={k}')

    return prev_classification, k


def apply_kmeans_to_masked(masked_array, k, seed=42):
    """applies k-means algorithm to masked array."""
    
    # compress array to contain only unmasked values
    spatial_mask = ~masked_array.mask.any(axis=0)
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
    # uncompressed_classifications = uncompress_cube(classifications[:, 0], pixel_indices, (h, w))
    uncompressed_classifications = uncompress_cube(classifications, pixel_indices, (h, w))

    return uncompressed_classifications


def get_roi(masked_arr):
    """places a rectangle for each center of mass"""

    center_of_mass = get_center_of_mass(masked_arr)

    # find the largest rectangle centered at this point
    left, top, right, bottom = largest_rect_around_center(masked_arr, center_of_mass)

    width = right - left + 1
    height = bottom - top + 1
    area = width * height

    rect = (left, top, width, height)

    return area, rect

