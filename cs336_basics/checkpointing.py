import torch 
import os
from typing import IO, BinaryIO

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint = dict()
    checkpoint['model'] = orig_model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['iteration'] = iteration

    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module | None = None, optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(src)

    if model is not None:
        model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint["iteration"]