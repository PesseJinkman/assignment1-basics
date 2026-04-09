import pickle
import time
import tracemalloc
from pathlib import Path

from train_bpe import train_bpe


def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def main() -> None:
    tracemalloc.start()
    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )
    output_path = Path("artifacts") / "owt_train_bpe.pkl"
    save_vocab_and_merges(vocab, merges, output_path)
    elapsed_time = time.perf_counter() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Saved tokenizer to {output_path}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Current memory usage: {current_memory / (1024 * 1024):.2f} MB")
    print(f"Peak memory usage: {peak_memory / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
