import tensorflow as tf
import numpy as np

import pickle
import zipfile
import collections
import time
from collections import Counter
# save dictionary as pickle
def dict_to_pickle(pickle_file, dict_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load pickle as dictionary
def dict_pickle_open(pickle_file):
    with open(pickle_file, 'rb') as handle:
        dictionary = pickle.load(handle)

    return dictionary

# preprocess the data by using tensorflow because of speed
def preprocess(file):
    with zipfile.ZipFile(file) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()

    return data

# make word list, dictionary and reverse dictionary
def word_dictionary(words):

    ### words : total words list of text8 dataset
    word_lst = [('UNK',0)]
    word_dict = dict()

    # total 100000 most common (word, frequency) pair
    word_lst.extend(Counter(words).most_common(99999))

    for idx, (word, _) in enumerate(word_lst):
        # dictionary : {'word' : 'index'}
        word_dict[word] = idx

    # reverse dictionary : {'index' : 'word'}
    word_reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_lst, word_dict, word_reverse_dict

# make words index list with original datset and count the unknown words

def unknown_count(words, word_lst, word_dict):
    words_index = list()
    unk_cnt = 0
    for word in words:
        if word in word_dict:
            idx = word_dict[word]
        else:
            idx = 0
            unk_cnt += 1 # 'UNK' count

        words_index.append(idx)

    word_lst[0] = ('UNK', unk_cnt)
    return words_index, word_lst

# Subsampling function
def Subsampling(word_lst, words_index, threshold, unigram):
    word_prob = dict()

    for idx, (word, freq) in enumerate(word_lst):
        #prob = freq / len(words_index)
        word_prob[idx] = 1 - np.sqrt(threshold / unigram[idx])

    return [idx for idx in words_index if word_prob[idx] < 1 - np.random.random()]

# Make Unigram Distribution
def Unigram_dict(words, word_lst):
    unigram = {}

    for idx, (word, freq) in enumerate(word_lst):
        unigram[idx] = freq / len(words)

    return unigram

def total_sampling (file_data, subsampling=False, threshold=None):
    start = time.time()

    word_lst, word_dict, word_reverse_dict = word_dictionary(file_data)   # 10만개 (word, freq) pair, word-index dict and reverse dict
    make_dict_time = time.time()

    words_index, word_lst = unknown_count(file_data, word_lst, word_dict) # words_index for subsampling, unknown count
    unk_count = time.time()

    unigram_dict = Unigram_dict(file_data, word_lst)
    unigram_time = time.time()

    if subsampling:
        words_index = Subsampling(word_lst, words_index, threshold, unigram_dict)
        subsample_time = time.time()


    print ('making dictionary : ', make_dict_time - start)
    print ('unknown count : ', unk_count - make_dict_time)
    print ('unigram dictionary : ', unigram_time - unk_count)
    print ('subsampling time : ', subsample_time - unigram_time)

    return file_data, word_lst, word_dict, word_reverse_dict, words_index, unigram_dict
