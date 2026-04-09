from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clip
from cs336_basics.learning_rate_schedule import lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "train_path": "data/train.bin",
        "val_path": "data/valid.bin",
        "dtype": "uint16",
    },
    "model": {
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,
    },
    "optimizer": {
        "lr": 3e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
    },
    "training": {
        "batch_size": 32,
        "max_iters": 10000,
        "warmup_iters": 1000,
        "lr_decay_iters": 10000,
        "min_lr": 3e-5,
        "eval_interval": 500,
        "eval_iters": 50,
        "log_interval": 10,
        "checkpoint_interval": 1000,
        "checkpoint_dir": "checkpoints",
        "resume_from": None,
        "compile": False,
        "seed": 1337,
        "device": "auto",
    },
    "wandb": {
        "enabled": True,
        "project": "cs336-basics",
        "run_name": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model from a JSON config.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to the training config JSON.")
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)

    return deep_update(DEFAULT_CONFIG, user_config)


def select_device(device_config: str) -> str:
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return device_config


def load_tokens(path: str | Path, dtype: str) -> np.ndarray:
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(f"Token dataset not found: {token_path}")

    if token_path.suffix == ".npy":
        tokens = np.load(token_path, mmap_mode="r")
    else:
        tokens = np.memmap(token_path, dtype=np.dtype(dtype), mode="r")

    if tokens.ndim != 1:
        raise ValueError(f"Expected a 1D token array, got shape {tokens.shape} from {token_path}")

    return tokens


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def batch_loss(model: torch.nn.Module, tokens: np.ndarray, batch_size: int, context_length: int, device: str) -> torch.Tensor:
    inputs, targets = get_batch(tokens, batch_size, context_length, device)
    logits = model(inputs)
    return cross_entropy(logits, targets)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for split, tokens in (("train", train_tokens), ("val", val_tokens)):
        split_losses = torch.empty(eval_iters)
        for i in range(eval_iters):
            split_losses[i] = batch_loss(model, tokens, batch_size, context_length, device).item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses


def init_wandb(config: dict[str, Any]):
    wandb_config = config["wandb"]
    if not wandb_config["enabled"]:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb.enabled is true, but wandb is not installed.") from exc

    return wandb.init(
        project=wandb_config["project"],
        name=wandb_config["run_name"],
        config=config,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_config = config["data"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    training_config = config["training"]

    device = select_device(training_config["device"])
    seed = training_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_tokens = load_tokens(data_config["train_path"], data_config["dtype"])
    val_tokens = load_tokens(data_config["val_path"], data_config["dtype"])

    model = TransformerLM(
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        rope_theta=model_config["rope_theta"],
        device=torch.device(device),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    )

    start_step = 0
    if training_config["resume_from"] is not None:
        start_step = load_checkpoint(training_config["resume_from"], model, optimizer)
        print(f"Resumed checkpoint from {training_config['resume_from']} at step {start_step}")

    if training_config["compile"]:
        model = torch.compile(model)

    model.train()
    run = init_wandb(config)
    checkpoint_dir = Path(training_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Training "
        f"on {device} for {training_config['max_iters']} steps "
        f"with {len(train_tokens):,} train tokens and {len(val_tokens):,} val tokens."
    )

    previous_time = time.perf_counter()
    for step in range(start_step, training_config["max_iters"]):
        lr = lr_cosine_schedule(
            step,
            alpha_min=training_config["min_lr"],
            alpha_max=optimizer_config["lr"],
            t_w=training_config["warmup_iters"],
            t_c=training_config["lr_decay_iters"],
        )
        set_learning_rate(optimizer, lr)

        loss = batch_loss(
            model,
            train_tokens,
            training_config["batch_size"],
            model_config["context_length"],
            device,
        )

        optimizer.zero_grad()
        loss.backward()

        grad_norm = None
        max_grad_norm = optimizer_config["max_grad_norm"]
        if max_grad_norm is not None:
            grad_norm = gradient_clip(model.parameters(), max_grad_norm).item()

        optimizer.step()

        completed_step = step + 1
        if completed_step % training_config["log_interval"] == 0:
            current_time = time.perf_counter()
            elapsed_ms = (current_time - previous_time) * 1000 / training_config["log_interval"]
            previous_time = current_time
            perplexity = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            log_items = {
                "step": completed_step,
                "train/loss": loss.item(),
                "train/perplexity": perplexity,
                "lr": lr,
                "ms_per_step": elapsed_ms,
            }
            if grad_norm is not None:
                log_items["grad_norm"] = grad_norm

            print(
                f"step {completed_step:6d} | loss {loss.item():.4f} | ppl {perplexity:.2f} | "
                f"lr {lr:.2e} | {elapsed_ms:.1f} ms/step"
            )
            if run is not None:
                run.log(log_items, step=completed_step)

        if completed_step % training_config["eval_interval"] == 0:
            eval_losses = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                training_config["batch_size"],
                model_config["context_length"],
                device,
                training_config["eval_iters"],
            )
            print(
                f"eval step {completed_step:6d} | train loss {eval_losses['train']:.4f} | "
                f"val loss {eval_losses['val']:.4f}"
            )
            if run is not None:
                run.log(
                    {
                        "eval/train_loss": eval_losses["train"],
                        "eval/val_loss": eval_losses["val"],
                    },
                    step=completed_step,
                )

        if completed_step % training_config["checkpoint_interval"] == 0:
            checkpoint_path = checkpoint_dir / f"ckpt_{completed_step:06d}.pt"
            save_checkpoint(model, optimizer, completed_step, checkpoint_path)
            save_checkpoint(model, optimizer, completed_step, checkpoint_dir / "latest.pt")
            print(f"saved checkpoint to {checkpoint_path}")

    final_path = checkpoint_dir / f"ckpt_{training_config['max_iters']:06d}.pt"
    save_checkpoint(model, optimizer, training_config["max_iters"], final_path)
    save_checkpoint(model, optimizer, training_config["max_iters"], checkpoint_dir / "latest.pt")
    print(f"Training complete. Final checkpoint saved to {final_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
