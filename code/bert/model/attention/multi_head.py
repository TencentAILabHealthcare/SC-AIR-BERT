import torch.nn as nn
from .single import Attention
import torch
import numpy as np


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # save the attention
        # attn_mean = torch.mean(attn, dim=0)
        # attn_sum = torch.sum(attn_mean,dim=1)
        # attn_cls =  attn_mean[:,0,:]
        # np.savetxt('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_finalVersion/result/heatmap/RAKFKQLL_attention.txt',attn_sum.detach().cpu().numpy(),fmt='%f',delimiter=',')
        # np.savetxt('/aaa/louisyuzhao/guy2/xiaonasu/ImmuneBLAST_finalVersion/result/heatmap/RAKFKQLL_cls_attention.txt',attn_cls.detach().cpu().numpy(),fmt='%f',delimiter=',')

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
