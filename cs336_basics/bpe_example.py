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

merges = 6
for _ in range(merges):

    count = defaultdict(int)
    for key, value in freq.items():
        for curr, next in zip(key, key[1:]):
            count[curr+next] += value

    merge_chars = max(count, key=lambda k: (count[k], k))
    
    new_freq = defaultdict(int)
    for key, value in freq.items():
        new_key = []
        i = 0
        while i < len(key):
            if i < len(key) - 1 and key[i] + key[i + 1] == merge_chars:
                new_key.append(merge_chars)
                i += 2
            else:
                new_key.append(key[i])
                i += 1

        new_freq[tuple(new_key)] += value

    freq = new_freq
    vocab.append(merge_chars)
    
print(vocab)