import torch
import json
import os

from torch.utils.data import Dataset, DataLoader

from utils.utils import load_json

#----------DATASET----------
class ViInforgraphicDataset(Dataset):
    def __init__(self, imdb_file_path):
        super().__init__()
        imdb = load_json(imdb_file_path)
        self.data = []
        for item in imdb:
            self.data.append({
                "id": item["image_id"],
                "im_width": item["image_width"],
                "im_height": item["image_height"],
                "ocr_tokens": item["ocr_tokens"],
                "ocr_boxes": item["ocr_normalized_boxes"],
                "obj_boxes": item["obj_normalized_boxes"],
            })


    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    list_id = [item["id"] for item in batch]
    list_im_width = [item["im_width"] for item in batch]
    list_im_height = [item["im_height"] for item in batch]
    list_ocr_tokens = [item["ocr_tokens"] for item in batch]
    list_ocr_boxes = [item["ocr_boxes"] for item in batch]
    list_obj_boxes = [item["obj_boxes"] for item in batch]
    return list_id, list_im_width, list_im_height, list_ocr_tokens, list_ocr_boxes, list_obj_boxes

def get_loader(config, imdb_path, shuffle=True):
    dataset = ViInforgraphicDataset(imdb_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader
