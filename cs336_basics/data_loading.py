import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start_idx = len(x) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(f"Input array length {len(x)} is too short for context_length {context_length}")

    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    x_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    y_sequences = np.zeros((batch_size, context_length), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        x_sequences[i] = x[start_idx : start_idx + context_length]
        y_sequences[i] = x[start_idx + 1 : start_idx + context_length + 1]

    x_batch = torch.from_numpy(x_sequences)
    y_batch = torch.from_numpy(y_sequences)

    if device.startswith("cuda"):
        # Pin memory if on GPU
        x_batch, y_batch = (
            x_batch.pin_memory().to(device, non_blocking=True),
            y_batch.pin_memory().to(device, non_blocking=True),
        )
    else:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    return x_batch, y_batch