import torch
import os
import numpy as np
from PIL import Image
from typing import List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from utils.utils import load_vocab, load_list_images_fast
from utils.module_utils import fasttext_embedding_module
from icecream import ic

class SaVOExtractor(nn.Module):
    """
        Salient Visual Object Extractor
    """
    def __init__(self, savo_config, model_clip, processor_clip, fasttext_model, hidden_size):
        super().__init__()
        self.savo_config = savo_config
        self.model      = model_clip
        self.processor  = processor_clip
        self.fasttext_model = fasttext_model
        self.top_k = savo_config["top_k"]

        # Load vocab and text prompts
        self.load_vocab(
            vocab_path_en=self.savo_config["vocab_path_en"],
            vocab_path_vi=self.savo_config["vocab_path_vi"],
        )

        # Layer embedding fasttext object concepts
        fasttext_dim = 300
        self.linear_voc = nn.Linear(
            in_features=fasttext_dim,
            out_features=hidden_size
        )
        # self.LayerNorm_voc = nn.LayerNorm(normalized_shape=fasttext_dim)
        self.LayerNorm_voc = nn.LayerNorm(normalized_shape=hidden_size)

        # Layer embedding object concepts score
        self.linear_score = nn.Linear(
            in_features=1,
            out_features=hidden_size
        )
        self.LayerNorm_score = nn.LayerNorm(normalized_shape=hidden_size)

        # Save loaded image data
        self.data_images = {}

    def load_vocab(self, vocab_path_en, vocab_path_vi):
        self.obj_vocab_vi = np.array(load_vocab(vocab_path_vi))
        self.obj_vocab_en = np.array(load_vocab(vocab_path_en))
        assert len(self.obj_vocab_vi) == len(self.obj_vocab_en)
        self.text_prompts = [f"A photo contain a {word}." for word in self.obj_vocab_en]


    def get_clip_similarty(self, image_paths, top_k):
        """
            :params image_paths:    BS, num_image_path
        """
        loaded_image_paths = [image_paths for image_paths in image_paths if image_paths in self.data_images.keys()]
        unloaded_image_paths = [image_paths for image_paths in image_paths if image_paths not in loaded_image_paths]
        
        # Load image from self.data_images
        loaded_images = [self.data_images[image_path] for image_path in loaded_image_paths]
        unloaded_images = load_list_images_fast(unloaded_image_paths, desc="Loading images CLIP similarity")
        images = loaded_images + unloaded_images
        
        # Save loaded images
        for path, image in zip(unloaded_image_paths, unloaded_images):
            self.data_images[path] = image
        # images  = [Image.open(image_path) for image_path in image_paths]
        inputs  = self.processor(text=self.text_prompts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs.to(self.model.device))
        logits_per_image = outputs.logits_per_image
        logits  = logits_per_image.softmax(dim=1)

        # Sort by top_K
        sorted_logits       = torch.sort(logits, dim=-1, descending=True)
        sorted_similarity   = sorted_logits.values
        sorted_indices      = sorted_logits.indices

        top_K_similarities  = sorted_similarity[:, :top_k] # BS, K
        top_K_sorted_indices  = sorted_indices[:, :top_k] # BS, K
        top_K_sorted_indices = top_K_sorted_indices.cpu().numpy()
        top_K_objects       = [self.obj_vocab_vi[sorted_indice] for sorted_indice in top_K_sorted_indices] # BS, K
        return top_K_similarities, top_K_objects
    

    def fasttext_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        fasttext_embedding = [
            fasttext_embedding_module(
                model=self.fasttext_model,
                word=word
            ) 
            for word in words
        ]
        return fasttext_embedding
    
    def forward(self, image_ids, image_dir):
        """
            :params image_ids:  BS, 1   : List of images
        """
        image_paths = [os.path.join(image_dir, f"{image_id}.png") for image_id in image_ids]
        top_K_similarities, top_K_objects = self.get_clip_similarty(
            image_paths=image_paths,
            top_k=self.top_k
        )

        #-- embedding of visual object concepts to Fasttext
        # BS, top_K, 300
        top_K_similarities = torch.tensor(top_K_similarities).unsqueeze(-1)
        top_K_fasttext_embeddings = [self.fasttext_embedding(top_K_object) for top_K_object in top_K_objects]
        ic(type(top_K_fasttext_embeddings[0]))
        top_K_fasttext_embeddings = torch.tensor(top_K_fasttext_embeddings).to(top_K_similarities.device)

        embed_voc = self.LayerNorm_voc(self.linear_voc(top_K_fasttext_embeddings))
        embed_score = self.LayerNorm_score(self.linear_score(top_K_similarities))
        
        visual_concept_embedding = embed_voc + embed_score
        
        # FT(a_k_voc ), x_k_voc
        # BS, k, 300 - BS, k, hidden_size 
        return top_K_fasttext_embeddings, visual_concept_embedding