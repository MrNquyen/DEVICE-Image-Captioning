import torch
import math
import numpy as np
from torch import nn

class SemanticAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size 
        # Layer
        self.q_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )

        self.k_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )

        self.v_linear = nn.Linear(
            in_features=input_size,
            out_features=input_size
        )
    
    def forward(self, Q_input, K_input, V_input):
        """
            :params Q_input:    BS, number_of_element_q , input_size
            :params K_input:    BS, number_of_element_k , input_size
            :params V_input:    BS, number_of_element_v , input_size
        """
        Q = self.q_linear(Q_input)
        K = self.k_linear(K_input)
        V = self.v_linear(V_input)
        QK = QK = torch.bmm(
            Q, torch.transpose(K, 2, 1), 
        ) # BS, number_of_element_q, number_of_element_k
        A = QK / math.sqrt(self.input_size)
        return A, V



class SemanticAttentionDeFUM(SemanticAttention):
    def __init__(self, hidden_size):
        super().__init__(self, hidden_size)


    def forward(self, apperance_embedding):
        """
            :params apperance_embedding:    BS, num_obj + num_ocr, hidden_size
        """
        Q = self.q_linear(apperance_embedding)  # BS, num_obj + num_ocr, hidden_size
        K = self.k_linear(apperance_embedding)  # BS, num_obj + num_ocr, hidden_size
        V = self.v_linear(apperance_embedding)  # BS, num_obj + num_ocr, hidden_size
        QK = QK = torch.bmm(
            Q, torch.transpose(K, 2, 1), 
        ) # BS, num_obj + num_ocr, num_obj + num_ocr
        A = QK / math.sqrt(self.input_size)
        return A, V
    


class SemanticAttentionSgAM(SemanticAttention):
    def __init__(self, fasttext_dim):
        super().__init__(self, fasttext_dim)


    def forward(self, fasttext_ocr_tokens, fasttext_object_concepts):
        """
            :params fasttext_ocr_tokens:        BS, num_ocr, 300
            :params fasttext_object_concepts:   BS, top_K, 300
        """
        Q = self.q_linear(fasttext_ocr_tokens)
        K = self.k_linear(fasttext_object_concepts)
        QK = QK = torch.bmm(
            Q, torch.transpose(K, 2, 1), 
        ) # BS, num_ocr, top_K
        A = QK / math.sqrt(self.input_size)
        return A



        
    
