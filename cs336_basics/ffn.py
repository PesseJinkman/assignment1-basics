import torch
from cs336_basics.linear import Linear

def silu_activation(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_hidden = silu_activation(self.W1.forward(x))
        gate_hidden = self.W3.forward(x)
        glu = silu_hidden*gate_hidden
        return self.W2.forward(glu)
