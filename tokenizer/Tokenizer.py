"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle any special tokens.
"""

from tokenizer.TokenizerBase import TokenizerBase, get_stats, merge
from multiprocessing import Pool, cpu_count
from itertools import repeat
from collections import Counter
import regex
import json

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def merge_string_key(key, pair, idx) -> tuple[str,str]:
    return (key, str(merge(json.loads(key), pair, idx)))

class Tokenizer(TokenizerBase):

    def __init__(self):
        super().__init__()

    def train(self, texts:list[str], vocab_size:int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # preproccesing, by applying regex and sorting out lists if length < 2
        ids_segments:dict[str,int] = {}
        for text in texts:
            for segment in regex.findall(GPT4_SPLIT_PATTERN, text):
                if len(segment) < 2: # no need to learn merging if len < 2 
                    continue
                text_bytes = segment.encode("utf-8") # raw bytes
                ids = list(text_bytes) # list of integers in range 0..255
                ids_segments[str(ids)] = ids_segments.get(str(ids), 0) + 1 # type: ignore

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # if there are no more merges possible, just run with out doing anything
            if len(ids_segments) < 1:
                continue
            # count up the number of times every consecutive pair appears
            if verbose:
                print("\t\tcounting")
            stats:dict[tuple[int,int], int] = {}
            with Pool(processes=cpu_count()-1) as pool:
                stats_list = pool.starmap(get_stats, map(lambda x : (json.loads(x), ids_segments[x], None), ids_segments.keys()))
            for stat in stats_list:
                stats = Counter(stats) + Counter(stat)
            # find the pair with the highest count
            pair:tuple[int,int] = max(stats, key=stats.get) # type: ignore
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            if verbose:
                print("\t\treplacing")
            with Pool(processes=cpu_count()-1) as pool:
                merged_keys = pool.starmap(merge_string_key, zip(ids_segments.keys(), repeat(pair), repeat(idx)))
            # clean segements of len < 2
            for mer in merged_keys:
                if mer[1] != mer[0]:
                    val = ids_segments.pop(mer[0])
                    if len(json.loads(mer[1])) > 1:
                        ids_segments[mer[1]] = val

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"\tmerge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids