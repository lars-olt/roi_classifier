import numpy as np

from marslab.imgops.masking import skymask, threshold_mask

def preprocess_cube(cube, homography_mask, shaddow_kwargs, skymask_kwargs):
    # perform preprocessing steps
    
    # get threshold masks
    shadow_tmask = threshold_mask(cube, **shaddow_kwargs)
    sky_tmask = skymask(cube, **skymask_kwargs)
    
    # combine threshold masks
    feature_mask = np.logical_or(shadow_tmask, sky_tmask)
    full_tmask = np.logical_or(feature_mask, homography_mask)
    
    # apply tmask to cube
    cube_preprocessed = mask_cube(cube, full_tmask)
    cube_preprocessed.mask = (
        cube_preprocessed.mask | ~np.isfinite(cube_preprocessed)
    )
    
    return cube_preprocessed


def mask_cube(cube, mask):
    """applies mask to the cube."""

    stacked_mask = np.repeat(mask[np.newaxis, :], cube.shape[0], axis=0)
    masked_cube = np.ma.masked_array(cube, mask=stacked_mask)

    return masked_cube