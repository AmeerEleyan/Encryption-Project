import glob
import os
import string
from random import shuffle
import re
import random

file_list = glob.glob(os.path.join(os.getcwd(), "./dataset/", "*.sgm"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())

corpus = corpus[0:10]


def clean_data(file_data):
    transtable = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    clean_words = file_data.translate(transtable)
    text = "".join([i for i in clean_words if not i.isdigit()])
    text = text.lower()
    text = re.sub('\\s+', ' ', text)
    return text


a_info = clean_data(corpus[0])
text_a_no_of_words = len(a_info)
text_a = a_info

b_info = clean_data(corpus[2])
text_b_no_of_words = len(b_info)
text_b = b_info

c_info = clean_data(corpus[3])
text_c_no_of_words = len(c_info)
text_c = c_info

list_of_final_data = []
SizeOfTamperedFile = (text_a_no_of_words + text_b_no_of_words + text_c_no_of_words) / 3
list_of_final_data.append(" ".join(random.choices(text_a, k = int(0.5 * SizeOfTamperedFile))))
list_of_final_data.append(" ".join(random.choices(text_b, k = int(0.3 * SizeOfTamperedFile))))
list_of_final_data.append(" ".join(random.choices(text_c, k = int(0.2 * SizeOfTamperedFile))))

TamperedFileNew = " ".join(list_of_final_data)


# print("The Tampered File New is")
# print(TamperedFileNew)


def shingle(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k + 1):
        shingle_set.append(text[i:i + k])
    return set(shingle_set)


def create_hash_func( vocab):
    # function for creating the hash vector/function
    hash_ex = list(range(1, len(vocab) + 1))
    shuffle(hash_ex)
    return hash_ex


def build_minhash_func(vocab_size: int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes


def create_hash(vector: list, minhash_func, vocab):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for func in minhash_func:
        for i in range(1, len(vocab) + 1):
            idx = func.index(i)
            signature_val = vector[idx]

            if signature_val == 1:
                signature.append(i)

    return signature


def split_vector(signature, b):
    assert len(signature) % b == 0, "the number of bands is not siutable"
    r = int(len(signature) / b)
    # code splitting signature in b parts
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i: i + r])
    return subvecs


# def split_vector(signature, b):
#
#     if (len(signature) != 0):
#         if (len(signature) % b == 0):
#             r = int(len(signature) / b)
#         # Splitting signature in b parts
#             subvecs = []
#             for i in range(0, len(signature), r):
#                 subvecs.append(signature[i: i + r])
#             return subvecs

def hash_bands(band, b):
    band_hash = []
    for i in band:
        summation = 0
        for j in i:
            summation = summation + j
        band_hash.append((3 * summation) % b)  # Any linear function to generate hash for each band
    return band_hash


# print(hash([3,5,7,9,2,6],3))
def calculate_similarity(list1, list2, b_size):
    count = 0
    for b, a in zip(list1, list2):
        if a == b:
            # print(a, b)
            count = count + 1
    return count / b_size


a_tamp_comp = []
b_tamp_comp = []
c_tamp_comp = []
counter = 0


def file_incorpus():
    for File_counter in corpus:
        File_counterCleanVersion = clean_data(File_counter)
        File_counter_shingle = shingle(" ".join(File_counterCleanVersion), 5)
        print(File_counter_shingle)


file_incorpus()


def LSH(k_value, minhash_vector_size, b_size):
    k = k_value
    TamperedFile_shingles = shingle(TamperedFileNew, k)
    vocab = TamperedFile_shingles
    counter = 0
    for File_counter in corpus:
        File_counterCleanVersion = clean_data(File_counter)
        File_cunter_shingle = shingle(" ".join(File_counterCleanVersion), k)
        vocab = vocab.union(File_cunter_shingle)  # generate a vocab by union all shingles togther

        counter += 1

    TamperedFileHotCoding = [1 if x in TamperedFile_shingles else 0 for x in vocab]
    minhash_func = build_minhash_func(len(vocab), minhash_vector_size, vocab)
    tamp_sig = create_hash(TamperedFileHotCoding, minhash_func, vocab)
    if len(tamp_sig) % b_size == 0:
        band_tamp = split_vector(tamp_sig, b_size)
        print(band_tamp)
        hash_band_tamp = hash_bands(band_tamp, b_size)
        print(hash_band_tamp)

    NewCounter = 0
    for File_NewCounter in corpus:
        FileNewCounter = clean_data(File_NewCounter)
        Text_NewCountShingle = shingle("".join(FileNewCounter), k)
        hot_NewCounter = [1 if x in Text_NewCountShingle else 0 for x in vocab]
        print(len(hot_NewCounter))
        NewCounter_sig = create_hash(hot_NewCounter, minhash_func, vocab)
        if (len(NewCounter_sig) != 0):
            if len(NewCounter_sig) % b_size == 0:
                band_NewCounter = split_vector(NewCounter_sig, b_size)
                hash_band_NewCounter = hash_bands(band_NewCounter, b_size)
                NewCounter_Similarity = calculate_similarity(hash_band_tamp, hash_band_NewCounter, b_size)
                a_percentage_value = abs(NewCounter_Similarity - 0.5)
                b_percentage_value = abs(NewCounter_Similarity - 0.3)
                c_percentage_value = abs(NewCounter_Similarity - 0.2)
                a_tamp_comp.append([k, minhash_vector_size, b_size, NewCounter_Similarity, a_percentage_value])
                b_tamp_comp.append([k, minhash_vector_size, b_size, NewCounter_Similarity, b_percentage_value])
                c_tamp_comp.append([k, minhash_vector_size, b_size, NewCounter_Similarity, c_percentage_value])
        NewCounter += 1;


# for i in range(2, 5):
#     for j in range(2, 5):
#         for k in range(2, 5):
#             LSH(i, j, k)
print("Please enter the value of shingle")
ShingleNo: int
ShingleNo = input();
print("Please enter the number of functions")
FunNo: int
FunNo = input()
print("Please enter the number of bands")
BandNo: int
BandNo = input()

# TamperedFile_shingles = shingle(TamperedFileNew, ShingleNo)
# vocab = TamperedFile_shingles
# counter = 0
# for File_counter in corpus:
#     File_counterCleanVersion = clean_data(File_counter)
#     File_cunter_shingle = shingle(" ".join(File_counterCleanVersion), ShingleNo)
#     vocab = vocab.union(File_cunter_shingle)  # generate a vocab by union all shingles togther
#
#     counter += 1
#
# TamperedFileHotCoding = [1 if x in TamperedFile_shingles else 0 for x in vocab]
# minhash_func = build_minhash_func(len(vocab), FunNo, vocab)
# tamp_sig = create_hash(TamperedFileHotCoding, minhash_func, vocab)
# Counter =0
# for File_NewCounter in corpus:
#     FileNewCounter = clean_data(File_NewCounter)
#     Text_NewCountShingle = shingle("".join(FileNewCounter), ShingleNo)
#     hot_NewCounter = [1 if x in Text_NewCountShingle else 0 for x in vocab]
#     print(len(hot_NewCounter))
#     NewCounter_sig = create_hash(hot_NewCounter, minhash_func, vocab)
#     if (len(NewCounter_sig) != 0):
#         count: int =0
#         for count in range (0,5):
#             if len(NewCounter_sig) % BandNo == 0:
#                 band_NewCounter = split_vector(NewCounter_sig, BandNo)
#                 hash_band_NewCounter = hash_bands(band_NewCounter, BandNo)
#                 break
#             else:
#                 print("Please enter a new value for bands numbers")
#                 BandNo= input()
# LSH(ShingleNo,FunNo, BandNo)
LSH(4, 5, 4)
minFile_i_Comp0 = min(a_tamp_comp, key = lambda x: x[4])
minFile_i_Comp1 = min(b_tamp_comp, key = lambda x: x[4])
minFile_i_Comp2 = min(c_tamp_comp, key = lambda x: x[4])
print("file number 1 participated in " + str(minFile_i_Comp0[3] * 100) + "% at k=" + str(
    minFile_i_Comp0[0]) + ", minhash_vector_size = " + str(minFile_i_Comp0[1]) + " and b_size = " + str(
    minFile_i_Comp0[2]))
print("file number 1 participated in " + str(minFile_i_Comp1[3] * 100) + "% at k=" + str(
    minFile_i_Comp1[0]) + ", minhash_vector_size = " + str(minFile_i_Comp1[1]) + " and b_size = " + str(
    minFile_i_Comp1[2]))
print("file number 1 participated in " + str(minFile_i_Comp1[3] * 100) + "% at k=" + str(
    minFile_i_Comp2[0]) + ", minhash_vector_size = " + str(minFile_i_Comp2[1]) + " and b_size = " + str(
    minFile_i_Comp2[2]))

# Why we cannot look at two featres in the same time
# A: 50, 20, 30
# B: 50, 20, 30
# C: 50, 20, 30

# ==> 20
