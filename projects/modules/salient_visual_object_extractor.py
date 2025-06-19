import torch
import os
import numpy as np
from PIL import Image
from typing import List
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from utils.utils import load_vocab
from utils.module_utils import fasttext_embedding_module

class SaVOExtractor(nn.Module):
    """
        Salient Visual Object Extractor
    """
    def __init__(self, model_clip, processor_clip, fasttext_model, hidden_size):
        super().__init__()
        self.model      = model_clip
        self.processor  = processor_clip
        self.fasttext_model = fasttext_model

        # Layer embedding fasttext object concepts
        fasttext_dim = 300
        self.linear_voc = nn.Linear(
            in_features=fasttext_dim,
            out_features=hidden_size
        )
        self.LayerNorm_voc = nn.LayerNorm(normalized_shape=fasttext_dim)

        # Layer embedding object concepts score
        self.linear_score = nn.Linear(
            in_features=1,
            out_features=hidden_size
        )
        self.LayerNorm_score = nn.LayerNorm(normalized_shape=1)


    def load_vocab(self, vocab_path_en, vocab_path_vi):
        self.obj_vocab_vi = np.array(load_vocab(vocab_path_vi))
        self.obj_vocab_en = np.array(load_vocab(vocab_path_en))
        assert len(self.obj_vocab_vi) == len(self.obj_vocab_en)
        self.text_prompts = [f"A photo contain a {word}." for word in self.obj_vocab_en]


    def get_clip_similarty(self, image_paths, top_k):
        """
            :params image_paths:    BS, num_image_path
        """
        images  = [Image.open(image_paths) for image_path in image_paths]
        inputs  = self.processor(text=self.text_prompts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits  = logits_per_image.softmax(dim=1)

        # Sort by top_K
        sorted_logits       = torch.sort(logits, dim=-1, descending=True)
        sorted_similarity   = sorted_logits.values
        sorted_indices      = sorted_logits.indices

        top_K_similarities  = sorted_similarity[:, :, :top_k] # BS, K
        top_K_objects       = [self.obj_vocab_vi[sorted_indice] for sorted_indice in sorted_indices] # BS, K
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
        image_paths = [os.path.join(image_dir, image_id) for image_id in image_ids]
        top_K_similarities, top_K_objects = self.get_clip_similarty(image_paths)

        #-- embedding of visual object concepts to Fasttext
        # BS, top_K, 300
        top_K_fasttext_embeddings = [self.fasttext_embedding(top_K_object) for top_K_object in top_K_objects]
        top_K_fasttext_embeddings = torch.tesnor(top_K_fasttext_embeddings)
        top_K_similarities = torch.tensor(top_K_similarities)

        embed_voc = self.LayerNorm_voc(self.linear_voc(top_K_fasttext_embeddings))
        embed_score = self.LayerNorm_score(self.linear_score(top_K_similarities))
        visual_concept_embedding = embed_voc + embed_score
        
        # FT(a_k_voc ), x_k_voc
        # BS, k, 300 - BS, k, hidden_size 
        return top_K_fasttext_embeddings, visual_concept_embedding