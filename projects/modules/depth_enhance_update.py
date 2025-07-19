import torch
import numpy as np
from torch import nn
from icecream import ic

from projects.modules.attention import SemanticAttentionDeFUM

class DeFUM(nn.Module):
    def __init__(self, feature_dim, defum_config):
        super().__init__()
        #-- Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=defum_config["nhead"],
            activation=defum_config["activation"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=defum_config["num_layers"])
        
        self.LayerNorm = nn.LayerNorm(normalized_shape=feature_dim)
        self.softmax = nn.Softmax(dim=-1)

        #-- Module
        self.semantic_attention = SemanticAttentionDeFUM(feature_dim)


    def concat(self, ocr_feats, obj_feats):
        """
            :params ocr_feats: BS, num_ocr, ocr_dim
            :params obj_feats: BS, num_obj, obj_dim
        """
        return torch.concat([ocr_feats, obj_feats], dim=1)
    

    def cal_relative_score(self, dv_i, dv_j):
        return max(dv_i, dv_j) / min(dv_i, dv_j)


    def cal_relative_depth_map(self, depth_value_visual_entity):
        """
            :params depth_value_visual_entity:  BS, num_obj + num_ocr, 1
        """
        relative_matrix = [
            [self.cal_relative_score(dv_i, dv_j) for dv_i in depth_value_visual_entity ]
            for dv_j in depth_value_visual_entity
        ]
        return torch.tensor(relative_matrix)


    def forward(self, ocr_feats, obj_feats, ocr_dvs, obj_dvs):
        """
            :params ocr_feats   : BS, num_ocr, ocr_dim (d)
            :params ocr_feats   : BS, num_obj, obj_dim (d)
            :params ocr_dvs     : BS, num_ocr, 1
            :params obj_dvs     : BS, num_obj, 1
        """
        num_ocr = ocr_feats.size(1)
        apperance_embeddings          = self.concat(ocr_feats, obj_feats) # BS, num_ocr + num_obj, d
        depth_value_visual_entities   = self.concat(ocr_dvs, obj_dvs)     # BS, num_ocr + num_obj, d
        # ic(f"Defum {apperance_embeddings}")
        # ic(f"Defum {ocr_feats}")
        # ic(f"Defum {obj_feats}")
        # Semantic Attention
        # ic(apperance_embeddings.device)
        # ic(depth_value_visual_entities.device)
        A, V = self.semantic_attention(apperance_embeddings)

        # Calculate relative depth map
        relative_depth_maps = []
        for depth_value_visual_entity in depth_value_visual_entities:
            relative_depth_map = self.cal_relative_depth_map(depth_value_visual_entity)
            relative_depth_maps.append(torch.log(relative_depth_map))
        # for item in relative_depth_maps:
        #     ic(item.shape)
        R = torch.stack(relative_depth_maps).to(torch.float32).to(A.device) # BS, num_ocr + num_obj, num_ocr + num_obj 

        # ic(A.device, V.device, R.device)
        # Transformers
        transformer_in = self.LayerNorm(
            apperance_embeddings + torch.bmm(
                self.softmax(A + R), V
            ) # Batch matrix multiplication
        )
        transformer_out = self.transformer_encoder(transformer_in) # BS, num_ocr + num_obj, d
        transformer_out = transformer_out[:, :num_ocr, :]
        return transformer_out # BS, num_ocr, d

        


    