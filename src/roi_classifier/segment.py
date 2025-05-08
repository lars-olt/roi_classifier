import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

def segment_img(model_path, img):
    
    # load SAM model
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=model_path)
    sam_model.to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # segment image
    output_mask = mask_generator.generate(img)
    
    # store segments in mask
    h, w, _ = img.shape
    segments = np.zeros((h, w))
    for i, val in enumerate(output_mask):
        mask = val["segmentation"]
        segments[mask] = i
        
    return segments
