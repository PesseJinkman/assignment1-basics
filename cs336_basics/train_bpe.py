import regex as re
import os
from typing import BinaryIO
from collections import defaultdict
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(args):
    input_path, start, end, special_tokens = args
    local_freq = defaultdict(int)
 
    with open(input_path, "rb") as f:
        f.seek(start)
        # Added replace() to be made usable on Windows
        chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n")
 
    pattern = "|".join(
        re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
    )
    chunk_splits = re.split(pattern, chunk)
 
    for chunk_split in chunk_splits:
        for match in re.finditer(PAT, chunk_split):
            token = tuple(bytes((b,)) for b in match.group().encode("utf-8"))
            local_freq[token] += 1
 
    return dict(local_freq)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    freq = defaultdict(int)


    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    
    for i in range(256):
        vocab[len(vocab)] = bytes((i,))

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

    jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_pretokenize_chunk, jobs)

    for local_freq in results:
        for key, value in local_freq.items():
            freq[key] += value
    
    count = defaultdict(int)
    pair_to_words = defaultdict(set)

    for key, value in freq.items():
        seen_pairs = set()
        for curr, next in zip(key, key[1:]):
            count[(curr, next)] += value
            seen_pairs.add((curr, next))

        for pair in seen_pairs:
            pair_to_words[pair].add(key)

    num_merges = vocab_size-(len(special_tokens)+256)
    for _ in range(num_merges):

        if not count:
            break

        merge_chars = max(count, key=lambda k: (count[k], k))
        affected_words = list(pair_to_words[merge_chars])

        updates = []

        for key in affected_words:
            value = freq[key]

            old_pairs = set(zip(key, key[1:]))
            for pair in zip(key, key[1:]):
                count[pair] -= value
                if count[pair] == 0:
                    del count[pair]

            for pair in old_pairs:
                pair_to_words[pair].discard(key)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]
            
            new_key = []
            i = 0
            while i < len(key):
                if i < len(key) - 1 and (key[i], key[i + 1]) == merge_chars:
                    new_key.append(key[i] + key[i + 1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_key = tuple(new_key)

            updates.append((key, new_key, value))

        for key, new_key, value in updates:
            del freq[key]
            freq[new_key] += value
        
        for key, new_key, value in updates:
            new_pairs = set(zip(new_key, new_key[1:]))
        
            for pair in zip(new_key, new_key[1:]):
                count[pair] += value
        
            for pair in new_pairs:
                pair_to_words[pair].add(new_key)

        vocab[len(vocab)] = (merge_chars[0]+merge_chars[1])
        merges.append(merge_chars)
        
    return vocab, merges

# if __name__ == "__main__":
#     vocab, merges = train_bpe("data/test.txt", 300, ['<|endoftext|>'])
