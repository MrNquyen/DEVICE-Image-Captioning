import json
import os
import numpy as np

from PIL import Image
from utils.utils import get_img_name
from collections import Counter

# https://github.com/Westlake-AGI-Lab/Distill-Any-Depth
class DepthExtractor():
    def __init__(self, depth_map_dir):
        list_depth_map_name = os.listdir(depth_map_dir)
        list_depth_map_path = [os.path.join(depth_map_dir, depth_map_name) for depth_map_name in list_depth_map_name]
        
        #-- id2depthmap
        self.data = {
            get_img_name(name): np.array(Image.open(depth_map_path))
            for name, depth_map_path in zip(list_depth_map_name, list_depth_map_path)
        }

    def get_depth_value(self, image_id, box):
        image = self.data[image_id]
        height, width, _ = image.shape
        
        # Load norm coordinate
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = box
        
        # Scale to original
        x_min = int(xmin_norm * width)
        y_min = int(ymin_norm * height)
        x_max = int(xmax_norm * width)
        y_max = int(ymax_norm * height)

        # Crop the image and get highest frequency gray value
        cropped_image = image[y_min:y_max, x_min:x_max]
        flatten_image = cropped_image.flatten()
        counter = Counter(flatten_image)
        highest_freq = counter.most_common()[0]
        return highest_freq / 255



