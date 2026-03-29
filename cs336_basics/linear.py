import math
import torch
from einops import einsum

class Linear(torch.nn.Module):
    
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
        self.W = torch.nn.Parameter(torch.nn.init.trunc_normal_(w, mean=0., std=std, a=-3*std, b=3*std))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")