from tokenizer import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_files('artifacts/tinystories_train_bpe_vocab.pkl', 'artifacts/tinystories_train_bpe_merges.pkl')

with open('data/TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8') as f:
    tokens = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

np.save('data/TinyStoriesV2-GPT4-train.npy', tokens)


