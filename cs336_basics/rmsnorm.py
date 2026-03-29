import torch
from einops import reduce

class RMSNorm(torch.nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_squared = reduce(x**2, "... d -> ... 1", "mean") + self.eps
        rms = torch.sqrt(rms_squared)
        rms_norm = (x/rms)*self.gain

        return rms_norm.to(in_dtype)



