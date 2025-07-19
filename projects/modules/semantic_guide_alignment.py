import torch 
import torch.nn.functional as F
from torch import nn
from typing import List

from utils.module_utils import fasttext_embedding_module
from projects.modules.salient_visual_object_extractor import SaVOExtractor
from projects.modules.attention import SemanticAttentionSgAM
from icecream import ic

class SgAM(nn.Module):
    """
        Semantic-guided Alignment Module
    """
    def __init__(self, sgam_config, model_clip, processor_clip, fasttext_model, hidden_size):
        super().__init__()
        self.sgam_config = sgam_config
        self.model      = model_clip
        self.processor  = processor_clip
        self.fasttext_model = fasttext_model
        self.salient_object_extractor = SaVOExtractor(
            savo_config=sgam_config["savo"],
            model_clip=model_clip, 
            processor_clip=processor_clip, 
            fasttext_model=fasttext_model, 
            hidden_size=hidden_size
        )


        # Layer
        fasttext_dim=300
        self.semantic_attention = SemanticAttentionSgAM(fasttext_dim=fasttext_dim)
    
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
    

    def forward(self, image_ids, image_dir, ocr_tokens):
        """
            :params ocr_tokens: BS, num_ocr : List of ocr_tokens of an image
            :params image_ids:  BS, 1       : List of images
        """
        top_K_fasttext_embeddings, visual_concept_embedding = self.salient_object_extractor(image_ids, image_dir)
        fasttext_ocr_tokens = [self.fasttext_embedding(tokens) for tokens in ocr_tokens]
        fasttext_ocr_tokens = torch.tensor(fasttext_ocr_tokens).to(top_K_fasttext_embeddings.device) # BS, num_ocr (M), 300

        # ic(fasttext_ocr_tokens.device)
        # ic(top_K_fasttext_embeddings.device)
        # Semantic attention
        Q_s = self.semantic_attention(
            fasttext_ocr_tokens=fasttext_ocr_tokens, 
            fasttext_object_concepts=top_K_fasttext_embeddings
        ) # BS, top K, num_ocr

        # Semantic Embedding
        Q_s = torch.bmm(Q_s, top_K_fasttext_embeddings)
        semantic_representation_ocr_tokens = F.normalize(fasttext_ocr_tokens + Q_s, p=2, dim=-1)
        return semantic_representation_ocr_tokens, visual_concept_embedding


