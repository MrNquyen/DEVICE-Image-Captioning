import json
import os
import numpy as np

from tqdm import tqdm
# from PIL import Image
import cv2
from utils.utils import get_img_name, load_img, load_list_images_fast, load_img_cache
from collections import Counter
from icecream import ic
# https://github.com/Westlake-AGI-Lab/Distill-Any-Depth
buffer = {}

class DepthExtractor():
    def __init__(self, depth_images_dir):
        list_depth_map_name = os.listdir(depth_images_dir)
        list_depth_map_path = [os.path.join(depth_images_dir, depth_map_name) for depth_map_name in list_depth_map_name]
        
        #-- id2depthmap
        self.data = {
            get_img_name(name): depth_map_path
            for name, depth_map_path \
                in tqdm(
                    zip(list_depth_map_name, list_depth_map_path),
                    desc="Loading depth image path"
                )
        }
        # depth_images = load_list_images_fast(list(self.data.values()), desc="Loading depth image")
        self.data_images = {}
        

    def get_depth_value(self, image_id, box):
        if image_id not in self.data_images:
            depth_map_path = self.data[image_id]
            depth_image = load_img_cache(depth_map_path)[:, :, 0]
            self.data_images[image_id] = depth_image
            image = np.array(depth_image)
        else:
            image = np.array(self.data_images[image_id])
        height, width = image.shape
        image = self.data_images[image_id]
        
        # Load norm coordinate
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = box
        
        # Scale to original
        x_min = int(xmin_norm * width)
        y_min = int(ymin_norm * height)
        x_max = int(xmax_norm * width)
        y_max = int(ymax_norm * height)

        x_min = min(x_min, x_max)
        x_max = max(x_min, x_max)
        y_min = min(y_min, y_max)
        y_max = max(y_min, y_max)

        if x_min == x_max:
            x_min -= 2
            x_max -= 1
        if y_min == y_max:
            y_min -= 2
            y_max -= 1


        if y_max > height:
            y_max = height
        if x_max > width:
            x_max = width
        
        if x_min > x_max:
            x_min, x_max = x_max, x_min
            y_min, y_max = y_max, y_min

        # Crop the image and get highest frequency gray value
        cropped_image = image[y_min:y_max, x_min:x_max]
        flatten_image = cropped_image.flatten()
        counter = Counter(flatten_image)
        highest_freq = None
        try:
            highest_freq = counter.most_common()[0][0]
        except:
            ic("-----")
            ic(image_id)
            ic(height, width)
            ic(box)
            ic(x_min, y_min, x_max, y_max)
            ic(flatten_image)
            ic(counter)
            ic(counter.most_common())
        # ic(highest_freq)
        return highest_freq / 255



