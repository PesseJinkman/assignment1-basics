import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def decode(
  model: TransformerLM,
  tokenizer: Tokenizer,
  prompt: str,
  max_length: int = 1024,
  temperature: float = 1.0,
  top_p: float = 0.9     
):
    
    end_id = tokenizer.encode("<|endoftext|>")[0]
    input_ids = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    context_length = model.context_length

    with torch.no_grad():
        for _ in range(max_length):
            input_ids_trunc = input_ids[-context_length:]
            input_tensor = torch.tensor(input_ids_trunc, device=device).unsqueeze(0)
            logits = model(input_tensor)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] &= ~sorted_indices_to_remove[..., :-1]
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[0, indices_to_remove] = 0
            probs /= probs.sum()

            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == end_id:
                break

            input_ids.append(next_id)
    
    return tokenizer.decode(input_ids)