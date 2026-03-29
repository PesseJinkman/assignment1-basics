import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x-torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(x)

    return exp/torch.sum(exp, dim=dim, keepdim=True)