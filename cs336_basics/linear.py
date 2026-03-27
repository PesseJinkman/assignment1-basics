import math
import torch
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        w = torch.empty((out_features, in_features), dtype=dtype, device=device)
        std = math.sqrt(2/(in_features+out_features))
        self.W = nn.Parameter(torch.nn.init.trunc_normal_(w, mean=0., std=std, a=-3*std, b=3*std))


    def forward(self, x):
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")