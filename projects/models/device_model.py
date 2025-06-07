import torch
import fasttext
import numpy as np
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from projects.modules.multimodal_embedding import ObjEmbedding, OCREmbedding
from projects.modules.multimodal_transformer import MultimodalTransformerEncoder, MultimodalTransformerDecoder


#---------- DYNAMIC POINTER NETWORK ----------
class DynamicPointerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self):
        NotImplemented


#---------- MODEL ----------
class BaseModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.build_model_init()

    def build_model_init(self):
        self.build_fasttext_model()
        self.model_clip()

    def model_clip(self):
        # "openai/clip-vit-large-patch14"
        self.model_clip = CLIPModel.from_pretrained(self.config["model_clip"])
        self.processor_clip = CLIPProcessor.from_pretrained(self.config["model_clip"])
    
    def build_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.config["fasttext_bin"])



#---------- DEVICE MODEL ----------
class DEVICE(BaseModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.obj_embedding = ObjEmbedding(
            config=self.config_model,
            device=self.device
        )
        self.ocr_embedding = OCREmbedding(
            model_clip=self.model_clip,
            processor_clip=self.processor_clip,
            config=self.config_model,
            device=self.device
        )

        self.multimodal_transformer_encoder = MultimodalTransformerEncoder(
            hidden_size=self.config["hidden_size"],
            mutimodal_transformer_config=self.config["mutimodal_transformer"]
        )

        self.multimodal_transformer_decoder = MultimodalTransformerDecoder(
            hidden_size=self.config["hidden_size"],
            mutimodal_transformer_config=self.config["mutimodal_transformer"]
        )


    def forward(
            self,
            image_ids,
            captions,
            ocr_tokens,
            ocr_feats,
            ocr_boxes,
            obj_feats,
            obj_boxes,
            ocr_conf
        ):
        """
            :params image_ids   :   BS, 1
            :params captions    :   BS, max_length
            :params ocr_tokens  :   BS, num_ocr, 1
            :params ocr_feats   :   BS, num_ocr, ocr_feat
            :params ocr_boxes   :   BS, num_ocr, 4
            :params ocr_conf    :   BS, num_ocr, 1
            :params obj_feats   :   BS, num_obj, ocr_feat
            :params obj_boxes   :   BS, num_obj, 4
        """
        obj_embedding = self.obj_embedding(
            list_image_id=image_ids,
            boxes=obj_boxes,
            obj_feat=obj_feats
        )

        ocr_embedding, semantic_representation_ocr_tokens, visual_concept_embedding = self.ocr_embedding(
            list_image_id=image_ids, 
            ocr_boxes=ocr_boxes, 
            ocr_feats=ocr_feats,
            ocr_tokens=ocr_tokens,
            ocr_conf=ocr_conf,
            obj_boxes=obj_boxes, 
            obj_feats=obj_feats
        )

        encoder_output = self.multimodal_transformer_encoder(
            obj_embedding=obj_embedding,
            ocr_embedding=ocr_embedding,
            semantic_representation_ocr_tokens=semantic_representation_ocr_tokens, 
            visual_concept_embedding=visual_concept_embedding,
        )

