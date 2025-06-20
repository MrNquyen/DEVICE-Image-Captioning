import torch
import fasttext
import math
import numpy as np
from torch import nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from projects.modules.multimodal_embedding import ObjEmbedding, OCREmbedding, Sync, WordEmbedding
from projects.modules.decoder import EncoderAsDecoder 
from utils.configs import Config
from utils.module_utils import _batch_padding, _batch_padding_string
from transformers import AutoTokenizer, AutoModel, AutoConfig, RobertaPreTrainedModel


#---------- DYNAMIC POINTER NETWORK ----------
class DynamicPointerNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self):
        NotImplemented


#---------- MODEL ----------
class BaseModel(nn.Module):
    def __init__(
            self, 
            config: Config, 
            device: str
        ):
        """
            :params config: Model Config
            :params device: device cuda
        """
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = self.config.config_model["hidden_size"]
        self.build_model_init()


    def build_model_init(self):
        # Finetune module is the module has lower lr than others module
        self.finetune_modules = []
        self.build_fasttext_model()
        self.model_clip()

    def add_finetune_modules(self, module: nn.Module):
        self.finetune_modules.append({
            'module': module,
            'lr_scale': self.config.config_optimizer["lr_scale"],
        })

    def model_clip(self):
        # "openai/clip-vit-large-patch14"
        self.model_clip = CLIPModel.from_pretrained(self.config["model_clip"])
        self.processor_clip = CLIPProcessor.from_pretrained(self.config["model_clip"])
    
    def build_fasttext_model(self):
        self.fasttext_model = fasttext.load_model(self.config["fasttext_bin"])



#---------- DEVICE MODEL ----------
class DEVICE(BaseModel):
    def __init__(self, config, device):
        """
            :params config: Model Config
            :params device: device cuda
        """
        super().__init__(config, device)
        self.build()


    #---- BUILD
    def build(self):
        self.build_model_params()
        self.build_layers()
        self.build_sync()
        self.load_pretrained()
        self.build_ouput()

    def build_model_params(self):
        self.ocr_config = self.config["ocr"]
        self.obj_config = self.config["obj"]
        self.num_ocr, self.num_obj = self.ocr_config["num_ocr"], self.obj_config["num_obj"]
        self.dim_ocr, self.dim_obj = self.ocr_config["dim_ocr"], self.obj_config["dim_obj"]
        self.feature_dim = self.config["feature_dim"]

    def build_sync(self):
        self.sync_ocr = Sync(
            in_dim=self.dim_ocr,
            out_dim=self.feature_dim
        )

        self.sync_obj = Sync(
            in_dim=self.dim_obj,
            out_dim=self.feature_dim
        )

    def build_layers(self):
        # Object embedding
        self.obj_embedding = ObjEmbedding(
            config=self.config,
            device=self.device
        )

        # OCR Embedding
        self.ocr_embedding = OCREmbedding(
            model_clip=self.model_clip,
            processor_clip=self.processor_clip,
            config=self.config,
            device=self.device
        )

        # Word Embedding
        self.word_embedding = WordEmbedding(
            model=self.model,
            tokenizer=self.tokenizer,
            text_embedding_config=self.config["text_embedding"]
        )

        # Pointer-wise network
        self.ocr_ptr_net = OcrPtrNet(
            hidden_size=self.hidden_size,
            query_key_size=self.word_embedding.common_vocab.get_size()
        )

    def build_decoder(self):
        self.decoder = EncoderAsDecoder(
            pretrained_model=self.pretrained_model,
            config=self.config.config_model["mutimodal_transformer"]
        )

    def build_ouput(self):
        # Num choices = num vocab
        num_choices = len(self.word_embedding.common_vocab.get_size())
        self.classifier = nn.Linear(
            in_features=self.hidden_size,
            out_features=num_choices
        )


    def load_pretrained(self):
        self.robertea_model_name = self.config.config_model["model_decoder"]
        roberta_config = AutoConfig(self.roberta_model_name)
        roberta_config.num_attention_heads = self.config["nhead"]
        roberta_config.num_hidden_layers = self.config["num_layers"]
        roberta_model = AutoModel.from_pretrained(
            self.roberta_model_name, 
            config=roberta_config
        )
        self.pretrained_model = roberta_model
        self.pretrained_tokenizer = AutoTokenizer(self.roberta_model_name)



    #---- HELPER FUNCTION
    def preprocess_batch(self, batch):
        """
            Function:
                - Padding ocr and obj to the same length
                - Create mask for ocr and obj
        """
        box_pad = torch.rand((1, 4))
        # Padding ocr
        ocr_feat_pad = torch.rand((1, self.dim_ocr))
        batch["list_ocr_boxes"], ocr_mask = _batch_padding(batch["list_ocr_boxes"], max_length=self.num_ocr, pad_value=box_pad)
        batch["list_ocr_feat"] = _batch_padding(batch["list_ocr_feat"], max_length=self.num_ocr, pad_value=ocr_feat_pad, return_mask=False)
        batch["list_ocr_tokens"] = _batch_padding_string(batch["list_ocr_tokens"], max_length=self.num_ocr, pad_value="<pad>", return_mask=False)
        batch["ocr_mask"] = ocr_mask

        # Padding obj
        obj_feat_pad = torch.rand((1, self.dim_obj))
        batch["list_obj_boxes"], obj_mask = _batch_padding(batch["list_obj_boxes"], max_length=self.num_obj, pad_value=box_pad)
        batch["list_obj_feat"] = _batch_padding(batch["list_obj_feat"], max_length=self.num_obj, pad_value=obj_feat_pad, return_mask=False)
        batch["obj_mask"] = obj_mask
        return batch


    def sync_ocr_obj(self, batch):
        """
            Sync OCR and OBJ features to the same feat_dim
        """
        batch["list_ocr_feat"] = self.sync_ocr(batch["list_ocr_feat"])
        batch["list_obj_feat"] = self.sync_obj(batch["list_obj_feat"])
        return batch
    

    def get_optimizer_parameters(self, config_optimizer):
        """
            -----
            Function:
                - Modify learning rate
                - Fine-tuning layer has lower learning rate than others
        """
        optimizer_param_groups = []
        base_lr = config_optimizer["params"]["lr"]
        scale_lr = config_optimizer["lr_scale"]
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * scale_lr
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})
        print(f"%=== Optimer params groups ===% \n {optimizer_param_groups}")
        return optimizer_param_groups

    # ---- FORWARD
    def forward_mmt(
            self,
            obj_embed,
            obj_mask,
            ocr_embed,
            ocr_mask,
            semantic_representation_ocr_tokens,
            visual_concept_embed,
            common_vocab_embed,
            prev_inds,
        ):
        """
            Forward batch to model
        """
        # -- Decoder
        mmt_results: dict  = self.decoder(
            obj_embed=obj_embed,
            obj_mask=obj_mask,
            ocr_embed=ocr_embed,
            ocr_mask=ocr_mask,
            ocr_semantic_embed=semantic_representation_ocr_tokens,
            visual_concept_embed=visual_concept_embed,
            common_vocab_embed=common_vocab_embed,
            prev_inds=prev_inds,
        )
        mmt_results["ocr_mask"] = ocr_mask
        return mmt_results
    

    def forward_output(self, results):
        mmt_dec_output = results['mmt_dec_output'] # BS, max_length, hidden_size
        mmt_ocr_output = results['mmt_ocr_output'] # BS, num_ocr, hidden_size
        ocr_mask = results['ocr_mask'] # BS, num_ocr


        fixed_scores = self.classifier(mmt_dec_output) #  BS, max_length, num_vocab
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        ) # BS, max_length, num_ocr
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1) # BS, max_length, num_vocab + num_ocr
        results['scores'] = scores
        return scores


    def forward(self, batch, get_prediction: bool):
        batch = self.preprocess_batch(batch)
        batch = self.sync_ocr_obj(batch)

        #-- OBJ Embedding
        obj_embed = self.obj_embedding(
            list_image_id=batch["list_id"], 
            boxes=batch["list_obj_boxes"], 
            obj_feats=batch["list_obj_feat"]
        )

        # -- OCR Embedding
        ocr_embed, \
        semantic_representation_ocr_tokens, \
        visual_concept_embed = self.ocr_embedding(
            list_image_id=batch["list_id"], 
            ocr_boxes=batch["list_ocr_boxes"], 
            ocr_feats=batch["list_ocr_feat"],
            ocr_tokens=batch["list_ocr_tokens"],
            ocr_conf=batch["list_ocr_scores"],
            obj_boxes=batch["list_obj_boxes"], 
            obj_feats=batch["list_obj_feat"]
        )

        # -- Common embed
        common_vocab_embed = self.word_embedding.common_vocab.get_vocab_embedding()
        
        # -- Training or Evaluating
        if self.training:
            #~ prev_inds
            prev_inds = self.word_embedding.get_prev_inds(
                sentences=batch["list_captions"],
                ocr_tokens=batch["list_ocr_tokens"]
            )
            results = self.forward_mmt(
                obj_embed=obj_embed,
                obj_mask=batch["obj_mask"],
                ocr_embed=ocr_embed,
                ocr_mask=batch["ocr_mask"],
                semantic_representation_ocr_tokens=semantic_representation_ocr_tokens,
                visual_concept_embed=visual_concept_embed,
                common_vocab_embed=common_vocab_embed,
                prev_inds=prev_inds
            )
            scores = self.forward_output(results)
        else:
            num_dec_step = self.word_embedding.max_length
            # Init prev_ids with <s> idx at begin, else where with <pad> (at idx 0)
            start_idx = self.word_embedding.common_vocab.get_start_index() 
            pad_idx = self.word_embedding.common_vocab.get_pad_index()
            batch_size = obj_embed.size(0)

            prev_inds = torch.full((batch_size, num_dec_step), pad_idx)
            prev_inds[:, 0] = start_idx
            scores = None
            for i in range(num_dec_step):
                results = self.forward_mmt(
                    obj_embed=obj_embed,
                    obj_mask=batch["obj_mask"],
                    ocr_embed=ocr_embed,
                    ocr_mask=batch["ocr_mask"],
                    semantic_representation_ocr_tokens=semantic_representation_ocr_tokens,
                    visual_concept_embed=visual_concept_embed,
                    common_vocab_embed=common_vocab_embed,
                    prev_inds=prev_inds
                )
                scores = self.forward_output(results)
                argmax_inds = scores.argmax(dim=-1)
                prev_inds = argmax_inds[:, :-1]
            return scores

# ----- DYNAMIC POINTER NETWORK -----
class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        """
            Parameters:
            ----------
                query_inputs    # BS, max_length, hidden_size
                key_inputs      # BS, num_ocr, hidden_size
                attention_mask  # BS, num_ocr
        """
        extended_attention_mask = attention_mask.squeeze(1)

        query_layer = self.query(query_inputs) # -> To vocab_size
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs) # -> To vocab_size

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


