# Created by Ameer Eleyan
# at 2/18/2023 11:29 AM

import glob
import os
import string
import re
import random
import LSH

file_list = glob.glob(os.path.join(os.getcwd(), "./dataset/", "*.sgm"))

corpus = []
count = 0
for file_path in file_list:
    if count == 100:
        break
    count += 1
    with open(file_path) as f_input:
        corpus.append(f_input.read())


def clean_data(file_data):
    transtable = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    clean_words = file_data.translate(transtable)
    text = "".join([i for i in clean_words if not i.isdigit()])
    text = text.lower()
    text = re.sub('\\s+', ' ', text)
    return text


a_info = clean_data(corpus[10])
text_a_no_of_words = len(a_info)
text_a = a_info

b_info = clean_data(corpus[40])
text_b_no_of_words = len(b_info)
text_b = b_info

c_info = clean_data(corpus[98])
text_c_no_of_words = len(c_info)
text_c = c_info

list_of_final_data = []
SizeOfTamperedFile = (text_a_no_of_words + text_b_no_of_words + text_c_no_of_words) / 3

# list_of_final_data.append(" ".join(text_a[0:int(0.5 * SizeOfTamperedFile )]))
# list_of_final_data.append(" ".join(text_b[0:int(0.3 * SizeOfTamperedFile )]))
# list_of_final_data.append(" ".join(text_c[0:int(0.2 * SizeOfTamperedFile )]))


list_of_final_data.append("".join(random.choices(text_a, k = int(0.5 * SizeOfTamperedFile))))
list_of_final_data.append("".join(random.choices(text_b, k = int(0.3 * SizeOfTamperedFile))))
list_of_final_data.append("".join(random.choices(text_c, k = int(0.2 * SizeOfTamperedFile))))

TamperedFileNew = " ".join(list_of_final_data)

print("The Tampered File New is")
print(TamperedFileNew)

a_tamp_comp = []
b_tamp_comp = []
c_tamp_comp = []


def ameer(k_value, minhash_vector_size, b_size):
    k = k_value
    TamperedFile_shingles = LSH.shingle(TamperedFileNew, k)
    text_a_shingles = LSH.shingle("".join(text_a), k)
    text_b_shingles = LSH.shingle("".join(text_b), k)
    text_c_shingles = LSH.shingle("".join(text_c), k)

    vocab = list(TamperedFile_shingles.union(text_a_shingles).union(text_b_shingles).union(
        text_c_shingles))  # generate a vocab by union all shingles togther

    # 1 hot encoding  for all documents

    tamp_1hot = [1 if x in TamperedFile_shingles else 0 for x in vocab]
    a_1hot = [1 if x in text_a_shingles else 0 for x in vocab]
    b_1hot = [1 if x in text_b_shingles else 0 for x in vocab]
    c_1hot = [1 if x in text_c_shingles else 0 for x in vocab]

    # we create 20 minhash vectors
    minhash_func = LSH.build_minhash_func(nbits = minhash_vector_size, vocab_length = len(vocab))

    # now create signatures
    tamp_sig = LSH.create_hash(tamp_1hot, minhash_func, vocab)
    a_sig = LSH.create_hash(a_1hot, minhash_func, vocab)
    b_sig = LSH.create_hash(b_1hot, minhash_func, vocab)
    c_sig = LSH.create_hash(c_1hot, minhash_func, vocab)

    if len(a_sig) % b_size == 0:
        band_tamp = LSH.split_vector(tamp_sig, b_size)

        band_a = LSH.split_vector(a_sig, b_size)
        band_b = LSH.split_vector(b_sig, b_size)
        band_c = LSH.split_vector(c_sig, b_size)

        hash_band_tamp = LSH.hash_bands(band_tamp, b_size)
        hash_band_a = LSH.hash_bands(band_a, b_size)
        hash_band_b = LSH.hash_bands(band_b, b_size)
        hash_band_c = LSH.hash_bands(band_c, b_size)

        a_tamp_sim = LSH.jaccard_similarity(hash_band_tamp, hash_band_a)
        b_tamp_sim = LSH.jaccard_similarity(hash_band_tamp, hash_band_b)
        c_tamp_sim = LSH.jaccard_similarity(hash_band_tamp, hash_band_c)

        a_percentage_value = abs(a_tamp_sim - 0.5)
        b_percentage_value = abs(b_tamp_sim - 0.3)
        c_percentage_value = abs(c_tamp_sim - 0.2)
        a_tamp_comp.append([k, minhash_vector_size, b_size, a_tamp_sim, a_percentage_value])
        b_tamp_comp.append([k, minhash_vector_size, b_size, b_tamp_sim, b_percentage_value])
        c_tamp_comp.append([k, minhash_vector_size, b_size, c_tamp_sim, c_percentage_value])


for i in range(2, 10):
    for j in range(2, 10):
        for k in range(2, 10):
            ameer(i, j, k)

minFile_i_Comp0 = min(a_tamp_comp, key = lambda x: x[4])
minFile_i_Comp1 = min(b_tamp_comp, key = lambda x: x[4])
minFile_i_Comp2 = min(c_tamp_comp, key = lambda x: x[4])
print("file  " + str(minFile_i_Comp0[3] * 100) + "% at k=" + str(minFile_i_Comp0[0]) + ", minhash_vector_size = " + str(
    minFile_i_Comp0[1]) + " and b_size = " + str(minFile_i_Comp0[2]))
print("file  " + str(minFile_i_Comp1[3] * 100) + "% at k=" + str(minFile_i_Comp1[0]) + ", minhash_vector_size = " + str(
    minFile_i_Comp1[1]) + " and b_size = " + str(minFile_i_Comp1[2]))
print("file  " + str(minFile_i_Comp2[3] * 100) + "% at k=" + str(minFile_i_Comp2[0]) + ", minhash_vector_size = " + str(
    minFile_i_Comp2[1]) + " and b_size = " + str(minFile_i_Comp2[2]))

# Why we cannot look at two featres in the same time
# A: 50, 20, 30
# B: 50, 20, 30
# C: 50, 20, 30

# ==> 20
