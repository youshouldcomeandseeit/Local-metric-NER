import math

import numpy as np
import torch.nn as nn
import torch
import tool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from transformers import BertModel
from sklearn.cluster import KMeans,SpectralClustering
from tool import tools
from torch.autograd import Variable

class _Encoder(nn.Module):
    def __init__(self, ent_type_size, encoder):
        super(_Encoder, self).__init__()
        self.ent_type_size = ent_type_size
        self.hidden_size = encoder.config.hidden_size
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        hidden = self.encoder(input_ids, attention_mask, token_type_ids)
        token_embeddings = torch.sum(torch.stack(hidden[2][-4:]), dim=0)
        return token_embeddings


class ellipsoid_NER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.encoder_path, output_hidden_states=True)
        self.dropout = nn.Dropout(config.dropout)
        self._encoder = _Encoder(config.ent_type_size, self.bert)
        self.ent_type_size = config.ent_type_size
        self.linear_size = config.linear_size
        self.head_ = nn.Linear(self.bert.config.hidden_size, self.linear_size)
        self.tail_ = nn.Linear(self.bert.config.hidden_size,  self.ent_type_size * self.linear_size)
        self.L_matric = nn.Parameter(torch.randn(self.ent_type_size, self.linear_size, config.in_dim))
        nn.init.xavier_uniform_(self.L_matric.data)
        # nn.init.kaiming_uniform_(self.L_matric.data)
        self.head_cls = nn.Linear(self.linear_size,self.ent_type_size)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device
        token_embeddings = self._encoder(input_ids, attention_mask, token_type_ids)  # [b,seq,dim]
        b, seq_len, _ = token_embeddings.shape
        head = self.head_(token_embeddings)
        tail = self.tail_(token_embeddings)

        head_ = torch.stack(torch.split(head, self.linear_size, dim=-1), dim=-2)   # [b,seq,1,dim]
        tail_ = torch.stack(torch.split(tail, self.linear_size, dim=-1), dim=-2)

        head_pos, tail_pos = tool.tools.rope(head_, tail_)

        head_L = torch.einsum('bihd, hdj -> bhij', head_pos, self.L_matric)  # [b,n,seq,dim']
        tail_L = torch.einsum('bihd, hdj -> bhij', tail_pos, self.L_matric)
        head2tail = torch.cdist(head_L,tail_L,p=2)
        sim = self.head_cls(head).transpose(-1,-2)

        reg = torch.norm(self.L_matric.unsqueeze(1) - self.L_matric.unsqueeze(0),dim=(2,3))

        return head2tail,sim,head













































