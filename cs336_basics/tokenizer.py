import pickle
import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inv = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}        
        self.special_tokens = special_tokens

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"

            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    next_id = len(self.vocab)
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
        else:
            self.special_tokens = None
            self.special_pattern = None

    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if self.special_pattern:
            text_splits = re.split(self.special_pattern, text)
        else:
            text_splits = [text]

        ids = []

        for text_split in text_splits:

            if self.special_tokens and text_split in self.special_tokens:
                ids.append(self.vocab_inv[text_split.encode("utf-8")])

            else:
                for match in re.finditer(PAT, text_split):
                    pre_token = [bytes((b,)) for b in match.group().encode("utf-8")]
                    while True:
                        best_rank = float("inf")
                        best_idx = None

                        # Scan adjacent pairs
                        for i in range(len(pre_token) - 1):
                            pair = (pre_token[i], pre_token[i + 1])
                            rank = self.merges_dict.get(pair)
                            if rank is not None and rank < best_rank:
                                best_rank = rank
                                best_idx = i

                        # If no merges found, we're done
                        if best_idx is None:
                            break

                        # Merge the best pair
                        merged = pre_token[best_idx] + pre_token[best_idx + 1]  # Concatenate bytes
                        pre_token = pre_token[:best_idx] + [merged] + pre_token[best_idx + 2 :]

                    token_ids = [self.vocab_inv[subword] for subword in pre_token]
                    ids.extend(token_ids)
                
        return ids
                

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")