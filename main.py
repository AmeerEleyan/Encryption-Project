# Created by Ameer Eleyan
# at 2/6/2023 10:00 PM

import glob
import os
import re
import random
from bs4 import BeautifulSoup, SoupStrainer
import LSH


def clean_data(file_data):
    text: str = ""
    for t in BeautifulSoup(file_data, "html.parser", parse_only = SoupStrainer('body')):
        text = str(t)
    text = text.replace("<body>", "")
    text = text.replace("Reuter", "")
    text = text.replace("</body>", "")
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub('\\s+', ' ', text)
    return text


def get_corpus():
    file_list = glob.glob(os.path.join(os.getcwd(), "./dataset/", "*.sgm"))
    corpus_dataset = []
    count = 0
    for file_path in file_list:
        if count == 1000:
            break
        count += 1
        with open(file_path) as f_input:
            try:
                corpus_dataset.append(clean_data(f_input.read()))
            except:
                pass
    return corpus_dataset


def generate_tampered_file():
    text_a = corpus[10]

    text_b = corpus[450]

    text_c = corpus[850]

    list_of_final_data = []
    size_of_tampered_file = (len(corpus[10]) + len(corpus[450]) + len(corpus[850])) / 3
    list_of_final_data.append("".join(random.choices(text_a, k = int(0.5 * size_of_tampered_file))))
    list_of_final_data.append("".join(random.choices(text_b, k = int(0.3 * size_of_tampered_file))))
    list_of_final_data.append("".join(random.choices(text_c, k = int(0.2 * size_of_tampered_file))))

    return " ".join(list_of_final_data)


def our_corpus(tampered_file, shingle_size, minhash_vector_size, band_size):
    global hash_band_tamp
    vocab = LSH.get_vocab(shingle_size, corpus, tampered_file)
    tampered_file_shingles = LSH.shingle(tampered_file, shingle_size)

    tampered_file_hot_coding = LSH.get_shingles_hot_encoding(tampered_file_shingles, vocab)
    minhash_func = LSH.build_minhash_func(nbits = minhash_vector_size, vocab_length = len(vocab))
    temp_signature = LSH.create_hash(tampered_file_hot_coding, minhash_func, vocab)

    if len(temp_signature) % band_size == 0:
        band_tamp = LSH.split_vector(temp_signature, band_size)
        hash_band_tamp = LSH.hash_bands(band_tamp, band_size)

    for data_corpus in corpus:
        new_shingle = LSH.shingle(data_corpus, shingle_size)
        data_corpus_hot = LSH.get_shingles_hot_encoding(new_shingle, vocab)
        data_corpus_signature = LSH.create_hash(data_corpus_hot, minhash_func, vocab)
        if len(data_corpus_signature) != 0:
            if len(data_corpus_signature) % band_size == 0:
                data_corpus_band = LSH.split_vector(data_corpus_signature, band_size)
                data_corpus_hash_band = LSH.hash_bands(data_corpus_band, band_size)
                data_corpus_similarity = LSH.jaccard_similarity(hash_band_tamp, data_corpus_hash_band)

                a_percentage_value = abs(data_corpus_similarity - 0.5)
                b_percentage_value = abs(data_corpus_similarity - 0.3)
                c_percentage_value = abs(data_corpus_similarity - 0.2)

                a_tamp_comp.append(
                    [shingle_size, minhash_vector_size, band_size, data_corpus_similarity, a_percentage_value])
                b_tamp_comp.append(
                    [shingle_size, minhash_vector_size, band_size, data_corpus_similarity, b_percentage_value])
                c_tamp_comp.append(
                    [shingle_size, minhash_vector_size, band_size, data_corpus_similarity, c_percentage_value])


a_tamp_comp = []
b_tamp_comp = []
c_tamp_comp = []

if __name__ == '__main__':
    corpus = get_corpus()
    tampered_file = generate_tampered_file()

    our_corpus(tampered_file, 2, 5, 5)
    minFile_i_Comp0 = min(a_tamp_comp, key = lambda x: x[4])
    minFile_i_Comp1 = min(b_tamp_comp, key = lambda x: x[4])
    minFile_i_Comp2 = min(c_tamp_comp, key = lambda x: x[4])
    print("file number 1 participated in " + str(minFile_i_Comp0[3] * 100) + "% at k=" + str(
        minFile_i_Comp0[0]) + ", minhash_vector_size = " + str(minFile_i_Comp0[1]) + " and b_size = " + str(
        minFile_i_Comp0[2]))
    print("file number 1 participated in " + str(minFile_i_Comp1[3] * 100) + "% at k=" + str(
        minFile_i_Comp1[0]) + ", minhash_vector_size = " + str(minFile_i_Comp1[1]) + " and b_size = " + str(
        minFile_i_Comp1[2]))
    print("file number 1 participated in " + str(minFile_i_Comp2[3] * 100) + "% at k=" + str(
        minFile_i_Comp2[0]) + ", minhash_vector_size = " + str(minFile_i_Comp2[1]) + " and b_size = " + str(
        minFile_i_Comp2[2]))