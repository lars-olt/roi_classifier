from .classifier import get_potential_rois
from .loader import load_cube, LoadResult
from .utils import mask_cube, average_spectra, cluster_spectra
from .display import plot_roi_img, plot_spectra
from marslab.imgops.masking import skymask, threshold_mask
from .segment import segment_img
from .constraints import WLS

import numpy as np

class ROIClassifier:
    def __init__(self):
        self.using_pixmaps = False
        self.load_result = None
        self.segments = None
        self.processed_data = None
        self.photometrically_calibrated = None
        self.unfiltered_roi = None
        self.roi_spectra = None
        self.roi_stds = None
        self.filtered_roi_indices = None
        self.results = None

    def load(self, LOAD_KWARGS):
        load_result: LoadResult = load_cube(**LOAD_KWARGS)
        self.load_result = load_result
        self.using_pixmaps = LOAD_KWARGS['do_apply_pixmaps']

    def preprocess(self, SHADDOW_KWARGS, SKYMASK_KWARGS, DO_APPLY_R_STAR):
        if self.load_result is None:
            raise ValueError("No data loaded.")
        
        cube = self.load_result['cube']
        
        if self.using_pixmaps:
            # use unmasked cube for shaddow/sky masking to avoid NaNs
            # NaNs are bad for masking :(
            base_bands = self.load_result['base_bands']
            cube_for_masking = np.array(
                [a for b, a in base_bands.items() if b.startswith('R')]
            )
        else:
            # cube does not contain NaNs when not using pixmap
            cube_for_masking = cube
        
        # get masks for shaddow and sky
        shadow_tmask = threshold_mask(cube_for_masking, **SHADDOW_KWARGS)
        sky_tmask = skymask(cube_for_masking, **SKYMASK_KWARGS)
        
        # combine threshold masks
        feature_mask = np.logical_or(shadow_tmask, sky_tmask)
        full_tmask = np.logical_or(feature_mask, self.load_result['homography_tmask'])
        
        # apply mask to cube
        cube_preprocessed = mask_cube(cube, full_tmask)
        cube_preprocessed.mask = (
            cube_preprocessed.mask | ~np.isfinite(cube_preprocessed)
        )
        
        self.processed_data = cube_preprocessed
        
        # apply photometric calibration
        # NOTE: using Data.metaget() here can be sketchy
        #  b/c ZCAM metadata has multiple solar elevation values
        #  in different coordinate systems, not always in the same
        #  order in different label versions!
        if DO_APPLY_R_STAR is True:
            meta = self.load_result["bandset"].metadata
            incidence = meta["INCIDENCE_ANGLE"].unique().mean()
            photometric_scaling = np.cos(incidence * 2 * np.pi / 360)
        else:
            photometric_scaling = 1

        self.photometrically_calibrated = cube_preprocessed / photometric_scaling
    
    def compute_potential_rois(self, MODEL_PATH, ROI_KWARGS):
        if self.processed_data is None:
            raise ValueError("No preprocessed data.")
        
        self.segments = segment_img(MODEL_PATH, self.load_result['rgb_img'])
        self.unfiltered_roi = get_potential_rois(self.segments, self.processed_data, **ROI_KWARGS)
        
    def filter_roi(self, ROI_AREA_THRESHOLD):
        if self.unfiltered_roi is None:
            raise ValueError("Compute roi before filtering.")
        
        # TODO
        # ignore roi below area threshold
        valid_rois = np.where([roi[2] * roi[3] >= ROI_AREA_THRESHOLD for roi in self.unfiltered_roi])[0]
        area_constrained_rois = np.array([self.unfiltered_roi[i] for i in valid_rois])
        
        # convert rect coords to plot coords
        plt_rois = []
        for i in range(len(area_constrained_rois)):
            x1, y1, w, h = area_constrained_rois[i]
            x2 = x1 + w
            y2 = y1 + h
            plt_rois.append((x1, y1, x2, y2))

        # calculate average spectra in roi regions
        self.roi_spectra, self.roi_stds = average_spectra(self.photometrically_calibrated, plt_rois)

        bayer_sorted_indices = np.argsort(WLS[:3])
        non_bayer_sorted_indices = np.argsort(WLS[3:]) + 3

        non_bayer_spectra = self.roi_spectra[:, non_bayer_sorted_indices]
        
        roi_spectra_norm = []

        for s in non_bayer_spectra:
            spectra_norm = s.copy()
            
            min = spectra_norm.min()
            spectra_norm -= min

            max = spectra_norm.max()
            spectra_norm /= max
            
            roi_spectra_norm.append(spectra_norm)

        roi_spectra_norm = np.array(roi_spectra_norm)
        
        classifications = cluster_spectra(roi_spectra_norm)
        
        # apply heuristic
        minimized_roi_indices = []

        areas = [coords[2] * coords[3] for coords in area_constrained_rois]  # pixel area of rois
        max_area_diff = np.max(areas) - np.min(areas)

        avg_albedos = np.mean(self.roi_spectra, axis=1)  # mean reflectance at each pixel
        avg_albedo = np.mean(avg_albedos)
        # max_albedo_diff = np.max(avg_albedos) - np.min(avg_albedos)

        avg_errs = np.mean(self.roi_stds, axis=1)
        max_err_diff = np.max(avg_errs) - np.min(avg_errs)

        # find most representative roi for each spectral category
        for category in np.unique(classifications):

            # indices of rois in current category
            indices = np.where(classifications == category)[0]

            curr_score = 0
            chosen_i = indices[0]
            for i in indices:
                curr_area = areas[i]
                area_norm = (curr_area - np.min(areas)) / max_area_diff
                
                curr_albedo = avg_albedos[i]
                albedo_norm = abs(curr_albedo - avg_albedo) / (np.max(avg_albedos) - avg_albedo)  # averaged albedo difference
                
                avg_err = avg_errs[i]
                error_norm = (avg_err - np.min(avg_errs)) / max_err_diff
                
                score = area_norm + albedo_norm - error_norm

                if score > curr_score:
                    curr_score = score
                    chosen_i = i

            minimized_roi_indices.append(chosen_i)
            
        self.filtered_roi_indices = minimized_roi_indices

    def show_rois(self):
        if self.filtered_roi_indices is None:
            raise ValueError("ROIs not filtered.")
        plot_roi_img(self.load_result['rgb_img'], self.unfiltered_roi[self.filtered_roi_indices])
    
    def show_spectra(self):
        if self.filtered_roi_indices is None:
            raise ValueError("ROIs not filtered.")
        plot_spectra(
            self.roi_spectra[self.filtered_roi_indices],
            self.roi_stds[self.filtered_roi_indices]
        )
