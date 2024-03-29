# Created by Ameer Eleyan
# at 2/6/2023 10:00 PM

import glob
import os
import re
import random
from bs4 import BeautifulSoup, SoupStrainer
import LSH
import csv


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


def get_corpus(no_files: int):
    file_list = glob.glob(os.path.join(os.getcwd(), "./dataset/", "*.sgm"))
    corpus_dataset = []
    count = 0
    for file_path in file_list:
        if count == no_files:
            break
        count += 1
        with open(file_path) as f_input:
            try:
                corpus_dataset.append(clean_data(f_input.read()))
            except:
                pass
    return corpus_dataset


def get_random_substring(s, percentage):
    if percentage < 0 or percentage > 1:
        raise ValueError("Percentage must be between 0 and 1")
    substring_length = int(len(s) * percentage)
    start_index = random.randint(0, len(s) - substring_length)
    return s[start_index:start_index + substring_length]


def generate_tampered_file(file_list: list):
    random.seed(10)
    tampered_files = ""
    for file_ in file_list:
        tampered_files += get_random_substring(s = corpus[file_[0]], percentage = file_[1])

    return tampered_files


files_I_details = []


def calculate_similarities(tampered_file, files, shingle_size, minhash_vector_size, band_size):
    global hash_band_tampered
    vocab = LSH.get_vocab(shingle_size = shingle_size,
                          corpus = corpus,
                          tampered_file = tampered_file)
    tampered_file_shingles = LSH.shingle(data = tampered_file,
                                         shingle_size = shingle_size)

    tampered_file_hot_coding = LSH.get_shingles_hot_encoding(shingles = tampered_file_shingles,
                                                             vocab = vocab)
    minhash_func = LSH.build_minhash_func(nbits = minhash_vector_size,
                                          vocab_length = len(vocab))
    tempered_signature = LSH.create_hash(vector = tampered_file_hot_coding,
                                         minhash_func = minhash_func,
                                         vocab = vocab)

    if len(tempered_signature) % band_size == 0:
        band_tampered = LSH.split_vector(signature = tempered_signature,
                                         band_size = band_size)
        hash_band_tampered = LSH.hash_bands(band = band_tampered,
                                            band_size = band_size,
                                            files_size = len(files))

    for i, data_corpus in zip(range(1, len(corpus)), corpus):
        new_shingle = LSH.shingle(data = data_corpus,
                                  shingle_size = shingle_size)
        data_corpus_hot = LSH.get_shingles_hot_encoding(shingles = new_shingle,
                                                        vocab = vocab)
        data_corpus_signature = LSH.create_hash(vector = data_corpus_hot,
                                                minhash_func = minhash_func,
                                                vocab = vocab)
        if len(data_corpus_signature) != 0:
            if len(data_corpus_signature) % band_size == 0:
                data_corpus_band = LSH.split_vector(signature = data_corpus_signature,
                                                    band_size = band_size)
                data_corpus_hash_band = LSH.hash_bands(band = data_corpus_band,
                                                       band_size = band_size,
                                                       files_size = len(files))
                data_corpus_similarity = LSH.jaccard_similarity(hash_band_tampered = data_corpus_hash_band,
                                                                data_corpus_hash_band = hash_band_tampered)

                current_file_details = [shingle_size, minhash_vector_size, band_size, i]
                for file_ in files:
                    current_file_details.append(abs(data_corpus_similarity - file_[1]))

                files_I_details.append(current_file_details)


if __name__ == '__main__':
    no_of_corpus_files = int(input("Enter number of files for corpus: "))

    no_of_files = int(input("Enter number of files for tampered: "))
    files = []
    for i in range(no_of_files):
        message = "Enter file ID(0-" + str(no_of_corpus_files) + ") and it percentage: "
        data = input(message).split()
        files.append([int(data[0]), float(data[1])])

    shingles = input("Enter shingle range: ").split()
    shingle_size_min = int(shingles[0])
    shingle_size_max = int(shingles[1])

    minhash_vector_size = input("Enter minhash_vector_size range: ").split()
    minhash_vector_size_min = int(minhash_vector_size[0])
    minhash_vector_size_max = int(minhash_vector_size[1])

    band_size = input("Enter band_size range: ").split()
    band_size_min = int(band_size[0])
    band_size_max = int(band_size[1])

    corpus = get_corpus(no_files = no_of_corpus_files)
    tampered_file = generate_tampered_file(file_list = files)

    print("**********")
    print("Wait!! Processing...")
    for shingle_size in range(shingle_size_min, shingle_size_max):
        for minhash_vector_size in range(minhash_vector_size_min, minhash_vector_size_max):
            for band_size in range(band_size_min, band_size_max):
                calculate_similarities(tampered_file = tampered_file,
                                       files = files,
                                       shingle_size = shingle_size,
                                       minhash_vector_size = minhash_vector_size,
                                       band_size = band_size)
    print("Finish")
    print("**********")
    output_file_name = input("Enter name for output file: ")
    csv_file = open('./' + output_file_name + '.csv', 'w', newline = "", encoding = "utf-8")
    header_list = ["File ID", "Percentage of participation", "Similarity rate", "Accuracy rate",
                   "Number of shingles", "Number of minhash size", "Number of bands"]
    csv_writer = csv.DictWriter(f = csv_file, fieldnames = header_list)
    csv_writer.writeheader()

    index = 4
    for file_ in files:
        max_file = max(files_I_details, key = lambda x: x[index])

        if max_file[index] >= file_[1]:
            temp_accuracy = file_[1] / max_file[index]
        else:
            temp_accuracy = max_file[index] / file_[1]
        accuracy = round(temp_accuracy * 100.0, 2)

        details = {"File ID": file_[0],
                   "Percentage of participation": str(round(file_[1] * 100.0, 2)) + "%",
                   "Similarity rate": str(round(max_file[index] * 100.0, 2)) + "%",
                   "Accuracy rate": str(accuracy) + "%",
                   "Number of shingles": max_file[0],
                   "Number of minhash size": max_file[1],
                   "Number of bands": max_file[2]}
        csv_writer.writerow(details)
        index += 1
    csv_file.close()
    print("**********")
    print("Open ", output_file_name + '.csv')
