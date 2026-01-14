import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):    
    def __init__(self, n_head, d_model, in_bias=True, out_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model*3, bias=in_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)
        self.n_head = n_head
        self.d_k = d_model // n_head
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if mask is not None:
            weight = weight.masked_fill(mask == 0, float('-inf'))
        weight = weight / (self.d_k ** 0.5)
        attn = F.softmax(weight, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        return out
    

class CrossAttention(nn.Module):
    def __init__(self, n_head, d_emb, d_cross, in_bias=True, out_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_emb, d_emb, bias=in_bias)
        self.k_proj = nn.Linear(d_cross, d_emb, bias=in_bias)
        self.v_proj = nn.Linear(d_cross, d_emb, bias=in_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=out_bias)
        self.n_head = n_head
        self.d_k = d_emb // n_head
    
    def forward(self, x, context):
        batch_size, seq_len, d_model = x.size()
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)
        q = q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        weight = weight / (self.d_k ** 0.5)
        attn = F.softmax(weight, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        return out
    
