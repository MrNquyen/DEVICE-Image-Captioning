## Base on paper description: Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA

import torch
import torch.nn.functional as F
from torch import nn

# MultimodalTransformer Base Model
class MultimodalTransformer(nn.Module):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__()
        self.hidden_size = hidden_size
        self.mutimodal_transformer_config = mutimodal_transformer_config
        self.max_length = mutimodal_transformer_config["max_length"]

    def forward():
        NotImplemented
        

# -- Multimodal Transformer Encoder
class MultimodalTransformerEncoder(MultimodalTransformer):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__(self, hidden_size, mutimodal_transformer_config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=mutimodal_transformer_config["nhead"],
            activation=mutimodal_transformer_config["activation"],
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=mutimodal_transformer_config["num_layers"])
        


    def forward(
            self, 
            obj_embedding,
            ocr_embedding,
            semantic_representation_ocr_tokens, 
            visual_concept_embedding,
        ):
        """
            :params obj_embedding:                      BS, num_obj, hidden_size
            :params ocr_embedding:                      BS, num_ocr, hidden_size
            :params semantic_representation_ocr_tokens: BS, num_ocr, hidden_size 
            :params visual_concept_embedding:           BS, top_K, hidden_size
            :params prev_word_embedding:                BS, hidden_size
        """
        concat_feature_embedding = torch.concat([
            obj_embedding,
            ocr_embedding,
            semantic_representation_ocr_tokens,
            visual_concept_embedding
        ], dim=1)
        encoder_output = self.transformer_encoder(concat_feature_embedding)
        # BS, num_obj + num_ocr + num_ocr + top_K, hidden_size 
        return encoder_output


# -- Multimodal Transformer Decoder
class MultimodalTransformerDecoder(MultimodalTransformer):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__(self, hidden_size, mutimodal_transformer_config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=mutimodal_transformer_config["nhead"],
            activation=mutimodal_transformer_config["activation"],
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=mutimodal_transformer_config["num_layers"])
        
    def forward(
            self,
            embedding_caption_token,
            encoder_output,
            prev_word_embedding
        ):
        """
            :params embedding_caption_token:    BS, num_ocr, 
            :params encoder_output:             BS, num_ocr, 
            :params prev_word_embedding:        BS, num_ocr, 
        """
        start_token = "<s>"
        for t in range(self.max_length):

        NotImplemented
        
        

