import torch
from torch import nn
from typing import List

from projects.modules.convert_depth_map import DepthExtractor
from projects.modules.depth_enhance_update import DeFUM
from projects.modules.semantic_guide_alignment import SgAM
from utils.module_utils import fasttext_embedding_module
from utils.phoc.build_phoc import build_phoc
from utils.vocab import PretrainedVocab, OCRVocab


#----------SYNC INPUT DIM----------
class Sync(nn.Module):
    # Paper required obj_feat and ocr_feat has the same d-dimension
    # Sync to one dimension (..., 1024, 2048, ...)
    # Init once and updating parameters
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.sync = nn.Linear(
            in_features=in_dim,
            out_features=out_dim
        )

    def forward(self, feats):
        """
            :params feats:   BS, num, original_feat_dim
        """
        return self.sync(feats)


#----------Word embedding----------
class WordEmbedding:
    def __init__(self, model, tokenizer, text_embedding_config):
        self.model = model
        self.tokenizer = tokenizer  
        self.config = text_embedding_config
        self.max_length = self.config["max_length"]

        vocab_path =self. config["common_vocab"]
        self.common_vocab = PretrainedVocab(
            model=self.model,
            tokenizer=self.tokenizer,
            vocab_file=vocab_path
        )
        
    def get_prev_inds(self, sentences, ocr_tokens):
        """
            Use to get inds of each token of the caption sentences

            Parameters:
            ----------
            sentences: List[str]
                - Caption of the images
            
            ocr_tokens: List[List[str]]

            Return:
            ----------
            prev_ids: Tensor:
                - All inds of all word in the sentences 
        """
        ocr_vocab_object = OCRVocab(ocr_tokens=ocr_tokens)

        start_token = self.common_vocab.get_start_token()
        end_token = self.common_vocab.get_end_token()
        sentences_tokens = [
            [start_token] + sentence.split(" ")[:self.max_length] + [end_token]
            for sentence in sentences
        ]

        # Get prev_inds
        prev_ids = [
            [
                self.common_vocab.get_size() + ocr_vocab_object[sen_id].get_idx_word[token]
                if token in ocr_tokens[sen_id]
                else ocr_vocab_object[sen_id].get_idx_word[token]
                for token in sentence_tokens
            ] 
            for sen_id, sentence_tokens in enumerate(sentences_tokens)
        ]
        return torch.tensor(prev_ids)




#----------Embedding----------
class BaseEmbedding(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.common_dim = config["feature_dim"]
        
        self.depth_extractor = DepthExtractor(depth_images_dir=config["depth_images_dir"])
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)


    def cal_depth_value(self, list_image_id, boxes):
        """
           :params list_image_id:   BS, 1           :Batch of image id 
           :params boxes        :   BS, num_obj, 4  :Batch of image id 
        """
        dv = []
        for image_id, im_boxes in zip(list_image_id, boxes): # List of image boxes in batch
            dv_i = []
            for im_box in im_boxes: # List of all boxes in image
                dv_ij = self.depth_extractor.get_depth_value(
                    image_id=image_id,
                    boxes=im_box
                )
                dv_i.append(dv_ij)
            dv.append(dv_i)
        
        dv = torch.tensor(dv)
        assert len(dv.shape()) == 3
        return dv


    def concat_depth_value(self, list_image_id, boxes):
        """
           :params list_image_id:   BS, 1           :Batch of image id 
           :params boxes        :   BS, num_obj, 4  :Batch of image id 
        """
        dv = self.cal_depth_value(list_image_id, boxes)
        boxes_3d  = torch.concat([boxes, dv], axis=-1)
        return boxes_3d 
    


#----------Embedding OCR----------
class ObjEmbedding(BaseEmbedding):
    def __init__(self, config, device):
        super().__init__(config, device)

        # Layer
        self.linear_feat = nn.Linear(
            in_features=self.common_dim,
            out_features=self.hidden_size
        )

        self.linear_box = nn.Linear(
            in_features=5,
            out_features=self.hidden_size
        )
    

    def forward(self, list_image_id, boxes, obj_feats):
        """
           :params list_image_id:   BS, 1                   :Batch of image id 
           :params boxes        :   BS, num_obj, 4          :Batch of obj boxes
           :params obj_feats    :   BS, num_obj, obj_dim (d):Batch of obj features 
        """
        boxes_3d = self.concat_depth_value(list_image_id, boxes)
        layer_norm_feat = self.LayerNorm(self.linear_feat(obj_feats))
        layer_norm_boxes = self.LayerNorm(self.linear_boxes(boxes_3d))
        obj_embed = layer_norm_feat + layer_norm_boxes
        return obj_embed


#----------Embedding OBJ----------
class OCREmbedding(BaseEmbedding):
    def __init__(self, model_clip, processor_clip, fasttext_model, config, device, **kwargs):
        super().__init__(config, device)
        self.image_dir = kwargs.get("image_dir", None)
        if self.image_dir == None:
            raise Exception("Image directory cannot be None")
        self.fasttext_model = fasttext_model
        # Layers
        self.linear_out_defum = nn.Linear(
            in_features=self.common_dim,
            out_features=self.hidden_size
        )

        fasttext_dim = 300
        self.linear_out_sgam = nn.Linear(
            in_features=fasttext_dim,
            out_features=self.hidden_size
        )
        
        phoc_dim = 604
        self.linear_out_phoc = nn.Linear(
            in_features=phoc_dim,
            out_features=self.hidden_size
        )

        self.linear_out_ocr_boxes = nn.Linear(
            in_features=5,
            out_features=self.hidden_size
        )

        self.linear_out_ocr_conf = nn.Linear(
            in_features=1,
            out_features=self.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(normalized_shape=self.hidden_size)
        
        # Modules
        self.DeFUM = DeFUM(
            hidden_size=self.hidden_size,
            defum_config=self.config["defum"]
        )
        self.SgAM = SgAM(
            model_clip=model_clip,
            processor_clip=processor_clip,
            fasttext_model=fasttext_model,
            hidden_size=config["hidden_size"]
        )


    def phoc_embedding(self, words: List[str]):
        """
            :params words:  List of word needed to embedded
        """
        phoc_embedding = [
            build_phoc(token=word) 
            for word in words
        ]
        return torch.tensor(phoc_embedding)
    

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
        return torch.tensor(fasttext_embedding)

    

    def forward(self, 
                list_image_id, 
                ocr_boxes, 
                ocr_feats,
                ocr_tokens,
                ocr_conf,
                obj_boxes, 
                obj_feats
            ):
        """
           :params list_image_id:   BS, 1                   :Batch of image id 
           :params ocr_boxes    :   BS, num_ocr, 4          :Batch of ocr boxes
           :params ocr_feats    :   BS, num_ocr, ocr_dim (d):Batch of ocr features 
           :params ocr_tokens   :   BS, num_ocr, 1          :Batch of ocr tokens 
           :params ocr_conf     :   BS, num_ocr, 1          :Batch of ocr confidence 
           :params obj_boxes    :   BS, num_obj, 4          :Batch of obj boxes
           :params obj_feats    :   BS, num_obj, obj_dim (d):Batch of obj features 
        """
        ocr_dvs = self.cal_depth_value(list_image_id, ocr_boxes)
        obj_dvs = self.cal_depth_value(list_image_id, obj_boxes)
        ocr_boxes = self.concat_depth_value(list_image_id=list_image_id, boxes=ocr_boxes)
        # -- Phoc embedding
        ocr_phoc_embedding = [self.phoc_embedding(tokens) for tokens in ocr_tokens]
        ocr_phoc_embedding = torch.tensor(ocr_phoc_embedding)
        
        # -- Defum
        depth_enhance_ocr_appearance = self.DeFUM(
            ocr_feats=ocr_feats, 
            obj_feats=obj_feats, 
            ocr_dvs=ocr_dvs, 
            obj_dvs=obj_dvs
        )

        # -- SgAM
        semantic_representation_ocr_tokens, visual_concept_embed = self.SgAM(
            image_ids=list_image_id, 
            image_dir=self.image_dir, 
            ocr_tokens=ocr_tokens
        )

        # -- OCR embedding
        # BS, num_ocr, hidden_size
        ocr_embed = self.LayerNorm(
            self.linear_out_defum(depth_enhance_ocr_appearance) + \
            self.linear_out_sgam(semantic_representation_ocr_tokens) + \
            self.linear_out_phoc(ocr_phoc_embedding)
        ) + \
        self.LayerNorm(
            self.linear_out_ocr_boxes(ocr_boxes) + \
            self.linear_out_ocr_conf(ocr_conf)
        )
        # x_ocr, x_ft', x_k_voc
        return ocr_embed, semantic_representation_ocr_tokens, visual_concept_embed
     