import copy
import torch.nn as nn
from .multihead_attention import MultiheadAttention


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RATransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, nrelation=0, dim_feedforward=2048, dropout=0.1, num_layers=3):
        super(RATransformerEncoder, self).__init__()
        encoder_layer = RATransformerEncoderLayer(d_model, nhead, nrelation, dim_feedforward, dropout)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, relation, mask=None, src_key_padding_mask=None):
        output = src
        for i in range(self.num_layers):
            output = self.layers[i](output, relation, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm:
            output = self.norm(output)
        return output


class RATransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nrelation=0, dim_feedforward=2048, dropout=0.1):
        super(RATransformerEncoderLayer, self).__init__()
        '''
        self.self_attn = nn.Sequential(
            MultiheadAttention(d_model, nhead, dropout=dropout),
            nn.Dropout(dropout)
        )
        '''
        # Multi-head Attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Relation Embeddings
        self.relation_k_emb = nn.Embedding(nrelation, self.self_attn.head_dim)
        self.relation_v_emb = nn.Embedding(nrelation, self.self_attn.head_dim)

    def forward(self, src, relation=None, src_mask=None, src_key_padding_mask=None):
        # Relation Embedding
        relation_k = self.relation_k_emb(relation) if relation is not None else None
        relation_v = self.relation_v_emb(relation) if relation is not None else None

        # self Multi-head Attention & Residual & Norm
        src2 = self.self_attn(src, src, src, relation_k=relation_k, relation_v=relation_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = self.dropout(src2)
        src = src + src2
        src = self.norm1(src)

        # FeedForward & Residual & Norm
        src2 = self.feed_forward(src)
        src = src + src2
        src = self.norm2(src)

        return src
