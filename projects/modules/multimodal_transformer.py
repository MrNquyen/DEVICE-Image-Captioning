## Base on paper description: Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA

import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.module_utils import _batch_gather, _get_causal_mask

# Base Model
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
            obj_embed,
            ocr_embed,
            semantic_representation_ocr_tokens, 
            visual_concept_embed,
        ):
        """
            :params obj_embed:                      BS, num_obj, hidden_size
            :params ocr_embed:                      BS, num_ocr, hidden_size
            :params semantic_representation_ocr_tokens: BS, num_ocr, hidden_size 
            :params visual_concept_embed:               BS, top_K, hidden_size
        """
        concat_feature_embedding = torch.concat([
            obj_embed,
            ocr_embed,
            semantic_representation_ocr_tokens,
            visual_concept_embed
        ], dim=1)
        encoder_output = self.transformer_encoder(concat_feature_embedding)
        # BS, num_obj + num_ocr + num_ocr + top_K, hidden_size 
        return encoder_output


# -- Previous Embedding
class PrevEmbedding(nn.Module):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__(self, hidden_size, mutimodal_transformer_config)
        self.DEC_LENGTH = mutimodal_transformer_config["max_length"] # Max caption output length
        self.TYPE_NUM = 2 # OCR or FROM_VOCAB
        self.hidden_size = hidden_size

        self.positional_embedding = nn.Embedding(
            num_embeddings=self.DEC_LENGTH, 
            embedding_dim=hidden_size
        )
        self.token_type_embedding = nn.Embedding(
            num_embeddings=self.TYPE_NUM, 
            embedding_dim=hidden_size
        )
        self.common_voc_embedding_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.ocr_embedding_norm = nn.LayerNorm(normalized_shape=self.hidden_size)
        self.emb_dropout = nn.Dropout(mutimodal_transformer_config.hidden_dropout_prob)


    def init_pe_weights(self):
        """
            Init weight for self.positional_embedding
        """
        # Create positional encoding sin cos from Attention Is All You Need! paper
        pe = torch.zeros(self.DEC_LENGTH, self.hidden_size)
        position = torch.arange(0, self.DEC_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float() *
            (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Assign custom weights
        self.positional_embedding.weight.data = pe.clone()


    def forward(
            self,
            common_voc_embed,
            ocr_embed,
            prev_ids
        ):
        """
            :params common_voc_embed    :  common_vocab_len, hidden_size:   All embedding of common vocab
            :params ocr_embed           :  BS, num_ocr, hidden_size     :   All ocr embedding
            :params prev_ids                :  BS, list_prev_idx_in_vocab   :   All idx in vocab of prev word in 
            ----
            Note:
                - Idx of ocr token start from: vocab_len + 1
                (Because append ocr_vocab to common_vocab)
                - When training, input all the gt caption and mask, so the model cannot see the future
            ----
            Function:
                - Lookup table embed position, and get prev embeded vector
        """
        # -- Params
        batch_size = prev_ids.shape[0]
        current_seq_length = prev_ids.shape[1]
        vocab_size = common_voc_embed.shape[0]
        ocr_size = ocr_embed.shape[1]

        # -- Get prev vector embed
        common_voc_embed = self.common_voc_embedding_norm(common_voc_embed)
        ocr_embed = self.ocr_embedding_norm(ocr_embed)
        assert common_voc_embed.size(-1) == ocr_embed.size(-1)

        common_voc_embed = common_voc_embed.unsqueeze(0).expand(batch_size, -1, -1)
        look_up_table_embed = torch.concat(
            [common_voc_embed, ocr_embed],
            dim=1
        )

        last_word_embeds = _batch_gather(
            x=look_up_table_embed, 
            inds=prev_ids
        )

        # -- Position 
        position_ids = torch.arange(
            current_seq_length,
            dtype=torch.long,
            device=ocr_embed.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.positional_embedding(position_ids)

        # -- Type embedding: 0: common tokens (False) - 1: ocr_tokens (True)
        type_ids = position_ids.ge(vocab_size).long()
        token_type_embeds = self.token_type_embedding(type_ids)

        # -- Position and token type
        pos_type_embeds = position_embeds + token_type_embeds 
        pos_type_embeds = self.emb_layer_norm(pos_type_embeds)
        pos_type_embeds = self.emb_dropout(pos_type_embeds)

        # -- LastWord, Position, token type
        prev_emb = last_word_embeds + pos_type_embeds
        return prev_emb # BS, num_prev_words, hidden_size



class MultimodalTransformerDecoder(MultimodalTransformer):
    def __init__(self, hidden_size, mutimodal_transformer_config):
        super().__init__(hidden_size, mutimodal_transformer_config)
        self.prev_embedding = PrevEmbedding(
            hidden_size=hidden_size, 
            mutimodal_transformer_config=mutimodal_transformer_config
        )

    def forward(
            self,
            encoder_input_embed,
            encoder_input_mask,
            ocr_emb,
            common_voc_emb,
            prev_inds
        ):
        
        prev_embed = self.prev_embedding(
            common_voc_embedding=common_voc_emb,
            ocr_embedding=ocr_emb,
            prev_ids=prev_inds,
        )

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask

        # BS, num_gt_token (input all the gt caption to decoder)
        attention_mask = torch.zeros(
            prev_embed.size(0),
            prev_embed.size(1),
            dtype=torch.float32,
            device=prev_embed.device
        )

        # Concat encoder_embed + prev_embed
        encoder_inputs = torch.cat(
            [encoder_input_embed, prev_embed],
            dim=1
        )

        # Concat encoder_embed_mask + prev_embed_mask
        encoder_inputs_mask = torch.cat(
            [encoder_input_mask, attention_mask],
            dim=1
        )

        # Offsets of each modality in the joint embedding space
        encoder_input_begin = 0
        encoder_input_embed_end = encoder_input_begin + encoder_input_embed.size(1)
        dec_input_begin = encoder_input_embed_end + 1
        dec_input_end = dec_input_begin + prev_embed.size(1)

        # Multihead broadcasting
        end_when_reach_maxlen = attention_mask.size(1) # 
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, end_when_reach_maxlen, 1
        )

        # Casual masking for multihead broadcasting
        # Start Casual masking at start of dec_input 
        extended_attention_mask[:, :, dec_input_end:, dec_input_end:] = \
            _get_causal_mask(dec_input_end - dec_input_begin, encoder_inputs.device)
        
        # Valid attention has value 0 - invalid -inf
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        NotImplemented