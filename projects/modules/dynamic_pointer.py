## Calculate the copy score for each ocr token 
## Bilinear interaction between 
## decoding output (z(t)) and each OCR token’s embedding output


import torch
from torch import nn

# From papper Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA
class DynamicPointerNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        #-- Layer
        self.linear_voc_embedding = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True
        )

        self.linear_ocr_embedding = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True
        )

        self.linear_out_decoder = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True
        )

    def forward(
            self,
            out_decoder,
            ocr_embedding 
        ):
        """
            :params out_decoder     :   BS, num_obj + num_ocr + num_ocr + top_K, hidden_size 
            :params ocr_embedding   :   BS, num_ocr, hidden_size
        """
        x = self.linear_ocr_embedding(ocr_embedding)
        y = self.linear_out_decoder(out_decoder)


        
