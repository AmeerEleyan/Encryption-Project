# Created by Ameer Eleyan
# at 2/17/2023 1:37 PM

from random import shuffle


def shingle(data: str, shingle_size: int):
    shingle_set = []
    for i in range(len(data) - shingle_size + 1):
        shingle_set.append(data[i:i + shingle_size])
    return set(shingle_set)


def create_hash_func(vocab_length: int):
    # function for creating the hash vector/function
    hash_ex = list(range(1, vocab_length + 1))
    shuffle(hash_ex)
    return hash_ex


def build_minhash_func(nbits: int, vocab_length: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_length = vocab_length))
    return hashes


def create_hash(vector: list, minhash_func, vocab):
    signature = []
    vocab_length = len(vocab)
    for func in minhash_func:
        for i in range(1, vocab_length + 1):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(i)
                break
    return signature


def split_vector(signature, b):
    assert len(signature) % b == 0, "the number of bands is not suitable"
    r = int(len(signature) / b)
    # code splitting signature in b parts
    sub_vecs = []
    for i in range(0, len(signature), r):
        sub_vecs.append(signature[i: i + r])
    return sub_vecs


def hash_bands(band, b):
    band_hash = []
    for i in band:
        summation = 0
        for j in i:
            summation = summation + j
        band_hash.append((3 * summation) % b)  # Any linear function to generate hash for each band
    return band_hash


def get_shingles_hot_encoding(shingles, vocab):
    return [1 if x in shingles else 0 for x in vocab]


def get_vocab(shingle_size, corpus, tampered_new_file):
    tampered_file_shingles = shingle(tampered_new_file, shingle_size)
    vocab = tampered_file_shingles
    for data in corpus:
        data_shingle = shingle("".join(data), shingle_size)
        vocab = vocab.union(data_shingle)  # generate a vocab by union all shingles together
    return vocab


def calculate_similarity(list1, list2, b_size):
    count = 0
    for b, a in zip(list1, list2):
        if a == b:
            count += 1
    return count / b_size


def jaccard_similarity(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))
