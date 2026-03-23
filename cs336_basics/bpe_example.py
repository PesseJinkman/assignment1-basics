from collections import defaultdict


corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

words = corpus.split()

vocab = []
vocab.append('<|endoftext|>')

for i in range(256):
    vocab.append(bytes((i,)))

freq = defaultdict(int)

for word in words:
    freq[tuple(bytes((b,)) for b in word.encode('utf-8'))] += 1

count = defaultdict(int)
pair_to_words = defaultdict(set)

for key, value in freq.items():
    seen_pairs = set()
    for curr, next in zip(key, key[1:]):
        count[(curr, next)] += value
        seen_pairs.add((curr, next))

    for pair in seen_pairs:
        pair_to_words[pair].add(key)

merges = 6
for _ in range(merges):

    if not count:
        break

    merge_chars = max(count, key=lambda k: (count[k], k[0]+k[1]))
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

    vocab.append(merge_chars[0]+merge_chars[1])
    
    
print(vocab)