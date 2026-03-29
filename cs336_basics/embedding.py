import torch
from einops import einsum

class Embedding(torch.nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:

        super().__init__()
        w = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        std = 1.0
        self.embedding = torch.nn.Parameter(torch.nn.init.trunc_normal_(w, mean=0., std=std, a=-3*std, b=3*std))    

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


