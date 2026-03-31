import torch

# @torch.compile
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    
    loss = torch.mean(-(target_logits-max_logits) + log_sum_exp)

    return loss
