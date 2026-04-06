import torch
from cs336_basics.embedding import Embedding
from cs336_basics.rope import RoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.linear import Linear

class TransformerLM(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: float = 10000.0,
        device : torch.device | None = None,
        dtype : torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        d_head = d_model // num_heads

        self.rope = RoPE(theta=rope_theta, d_k=d_head, max_seq_len=context_length, device=device, dtype=dtype)

        self.transformer_layers = torch.nn.ModuleList(
            [TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device, dtype=dtype) for _ in range(num_layers)]
        )

        self.rmsnorm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.context_length})")
        
        x = self.token_embeddings(x)

        for layer in self.transformer_layers:
            x = layer(x, self.rope)

        x = self.rmsnorm_final(x)
        x = self.lm_head(x)

        return x


