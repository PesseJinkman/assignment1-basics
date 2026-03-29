import torch
from einops import einsum
import math
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = q.shape[-1]
    
    attention_scores = einsum(q, k, "... seq_q d, ... seq_k d -> ... seq_q seq_k")/math.sqrt(d_k)
    attention_scores = torch.where(mask, attention_scores, float('-inf'))

    attention_weights = softmax(attention_scores, dim=-1)

    return einsum(attention_weights, v, "... seq_q seq_k, ... seq_k d -> ... seq_q d")


