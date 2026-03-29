import torch
from einops import rearrange
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class CausalMultiHeadSelfAttention(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        
        assert d_model%num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.W_qkv = Linear(in_features=d_model, out_features=3*d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.W_qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        y = scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)
        y = rearrange(y, "b h s d -> b s (h d)")

        return self.output_proj(y)