import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embed, n_token):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_emb):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_emb)
        self.attn = SelfAttention(n_head=n_head, d_model=n_emb)
        self.norm2 = nn.LayerNorm(n_emb)
        self.linear1 = nn.Linear(n_emb, n_emb * 4)
        self.linear2 = nn.Linear(n_emb * 4, n_emb)
        
    def forward(self, x):
        residual = x
        attn_out = self.attn(self.norm1(x), mask=True)
        x = attn_out + residual
        residual = x
        x = self.linear1(self.norm2(x))
        x = x * torch.sigmoid(1.702 *x)
        x = self.linear2(x)        
        return x