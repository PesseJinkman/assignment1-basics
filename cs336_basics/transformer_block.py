import torch
from cs336_basics.multihead_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.ffn import SwiGLU
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RoPE

class TransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mhsa = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, rope : RoPE | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        
        x = x + self.mhsa(self.rmsnorm1(x), rope, token_positions)
        x = x + self.ffn(self.rmsnorm2(x))

        return x

