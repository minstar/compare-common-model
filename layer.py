import numpy as np

import math
import time

from train_util import *
# Skip gram model
# CBOW model
# softmax model
# Negative Sampling Table
# hierarchical model

"""def generate_batch(words, index, window_size):
    start = index - window_size if (index - window_size) > 0 else 0
    stop = index + window_size
    # print (window_size, start, stop)
    target_words = set(words[start:index] + words[index+1 : stop+1])
    contextWords = list(target_words)
    currentWord = words[index]

    return currentWord, contextWords"""

def generate_batch(words, index, window_size):
    currentWord = words[index]
    contextWords = []

    if (index >= window_size) & (window_size+index < len(words)):
        contextWords = np.concatenate((words[index-window_size:index],words[index+1:index+window_size+1]))
    elif index < window_size:
        contextWords = np.concatenate((words[0:index],words[index+1:index+window_size+1]))
    else:
        contextWords = np.concatenate((words[index - window_size:index],words[index+1:]))
    contextWords = contextWords.astype(np.int64)

    return currentWord, contextWords

def Sigmoid(x):
    # clip input to -150~150
    x = np.clip(x, -150, 150)
    return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.))

def Skipgram(currentWord, contextWord, inputVectors, outputVectors,
             use_loss='Softmax', sample_table=None, encoder=None):
    """
    currentWord    : center word String
    contextWords   : list of string, which has 2 * C words -> one word each time
    inputVectors   : access by row number, such as inputVectors[5]...
    outputVectors  : access by row number same as inputVectors
    use_loss       : Softmax, Hierarchical Softmax, etc...
    """
    loss = 0.0
    N, V = inputVectors.shape

    hidden = inputVectors[currentWord] # current word to hidden vector

    if use_loss == 'Softmax':
        loss, grads, grads_idx = softmax(contextWord, currentWord, inputVectors, hidden, outputVectors)
    elif use_loss == 'Hierarchical_Softmax':
        loss, grads, grads_idx = hierarchical_softmax(contextWord, currentWord, inputVectors,
                                                      hidden, outputVectors, huffman_encoding=encoder)
    elif use_loss == 'Negative_Sampling':
        # get unigram distribution and negative sampling index
        neg_sample = sample_table.sampling(contextWord)
        loss, grads, grads_idx = negative_sampling(contextWord, currentWord, inputVectors,
                                                   hidden, outputVectors, samples=neg_sample)

    return loss, grads, grads_idx

def CBOW(currentWord, contextWords, inputVectors, outputVectors,
             use_loss='Softmax', sample_table=None, encoder=None):
    """
    currentWord    : center word String
    contextWords   : list of string, which has 2 * C words -> one word each time
    inputVectors   : access by row number, such as inputVectors[5]...
    outputVectors  : access by row number same as inputVectors
    use_loss       : Softmax, Hierarchical Softmax, etc...
    """
    loss = 0.0
    N, V = inputVectors.shape
    C = len(contextWords)

    hidden = np.zeros(V)
    for contextWord in contextWords:
        #contextWord = float(contextWord)
        hidden += inputVectors[contextWord]

    hidden /= C

    if use_loss == 'Softmax':
        loss, grads, grads_idx = softmax(currentWord, contextWords, inputVectors, hidden, outputVectors)
    elif use_loss == 'Hierarchical_Softmax':
        loss, grads, grads_idx = hierarchical_softmax(currentWord, contextWords, inputVectors,
                                                     hidden, outputVectors, huffman_encoding=encoder)
    elif use_loss == 'Negative_Sampling':
        # get unigram distribution and negative sampling index
        neg_sample = sample_table.sampling(currentWord)
        loss, grads, grads_idx = negative_sampling(currentWord, contextWords, inputVectors,
                                                   hidden, outputVectors, samples=neg_sample)

    return loss, grads, grads_idx

def softmax(currentWord, contextWord, inputVectors, hiddenVectors, outputVectors):
    loss = 0.0
    grads = {}
    grads_idx = {}

    out = np.dot(hiddenVectors, outputVectors.T)
    exp_out = np.exp(out)
    sum_out = exp_out / np.sum(exp_out)
    pred = sum_out[currentWord]

    # loss compute
    loss += -np.log(pred)

    # gradient compute
    grad_out = sum_out.copy()
    grad_out[currentWord] -= 1

    #print (grad_out.shape)
    grads['out'] = np.dot(grad_out.reshape(-1, 1), hiddenVectors.reshape(1, -1))
    grads['in'] = np.dot(outputVectors.T, grad_out)

    grads_idx['out'] = np.arange(outputVectors.shape[0])
    grads_idx['in']  = contextWord

    return loss, grads, grads_idx

def negative_sampling(target, input_word, inputVectors, hiddenVectors, outputVectors, samples):
    loss = 0.0
    grads = {}
    grads_idx = {}

    #print ('inputVectors shape ', inputVectors.shape)
    #print ('hiddenVectors shape ', hiddenVectors.shape)
    #print ('outputVectors shape ', outputVectors.shape)
    #print ('target word shape ', target.shape)
    #print ('input_word shape ', input_word.shape)

    true_word = outputVectors[target]
    nega_word = outputVectors[samples]

    true_out = np.dot(true_word, hiddenVectors)    # (1, 6) (6, )
    nega_out = np.dot(nega_word, hiddenVectors)    # (N, 6) (6, )

#    print ('true_out shape and nega_out shape')
#    print (true_out.shape, nega_out.shape)
    true_loss = Sigmoid(true_out)
    nega_loss = Sigmoid(-nega_out)

    # loss compute
    loss = -np.log(true_loss) - np.sum(np.log(nega_loss))

    # gradient compute
    true_grad = true_loss - 1  # sigmoid
    nega_grad = 1 - nega_loss  # sigmoid

    grads['true_out'] = np.multiply(true_grad, hiddenVectors)
    grads['nega_out'] = np.dot(nega_grad.reshape(-1, 1), hiddenVectors.reshape(1,-1))
    grads['true_in']  = np.multiply(true_word.T, true_grad)
    grads['nega_in']  = np.dot(nega_word.T, nega_grad)

    grads_idx['true_out'] = target
    grads_idx['nega_out'] = samples
    grads_idx['true_in']  = input_word
    grads_idx['nega_in']  = input_word

    return loss, grads, grads_idx

def hierarchical_softmax(currentWord, contextWord, inputVectors, hiddenVectors, outputVectors, huffman_encoding=None):
    loss = 0.0
    grads = {}
    grads_idx = {}

    target = huffman_encoding.code(currentWord)
    left, right = list(), list()
    idx = 0

    for string in target:
        if string == '0': # left
            node_lst, node_dict, reverse_node_dict = huffman_encoding.make_node_dict()
            left.append(node_dict[idx])
            idx = idx * 2 + 1
        elif string == '1':
            node_lst, node_dict, reverse_node_dict = huffman_encoding.make_node_dict()
            right.append(node_dict[idx])
            idx = idx * 2 + 2

    true_out = np.dot(hiddenVectors, outputVectors[left].T)
    nega_out = np.dot(hiddenVectors, outputVectors[right].T)

    true_prob = Sigmoid(true_out)
    nega_prob = Sigmoid(-nega_out)

    true_product = np.product(true_prob)
    nega_product = np.product(nega_prob)

    pred = true_product * nega_product
    loss += -np.log(pred)

    # compute gradient
    true_grad = true_product * (1 - 1 / true_prob)
    nega_grad = nega_product * (1 / nega_prob - 1)

    grads['true_out'] = np.dot(true_grad.reshape(-1, 1), hiddenVectors.reshape(1, -1))
    grads['nega_out'] = np.dot(nega_grad.reshape(-1, 1), hiddenVectors.reshape(1, -1))
    grads['true_in']  = np.dot(true_grad.T, outputVectors[left])
    grads['nega_in']  = np.dot(nega_grad.T, outputVectors[right])

    grads_idx['true_out'] = left
    grads_idx['nega_out'] = right
    grads_idx['true_in']  = contextWord
    grads_idx['nega_in']  = contextWord

    return loss, grads, grads_idx

# reference by github page
# Unigram Table : https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py
class Unigram_Table:
    #--- 3/4 power of unigram distribution selected by mikolov et al. 2013
    #--- make table and recall with index
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab, count, unigram_dictionary):
        self.count = count
        #vocab_size = len(vocab)
        power = 0.75
        table_size = int(1e8) # unigram table length
        table = np.zeros(table_size, np.uint32)

        norm_val = sum(math.pow(uni_prob, power) for uni_prob in unigram_dictionary.values())
        #norm = sum([math.pow(prob, power) for prob in unigram_dictionary])
        print ('Filling Unigram Table')
        p = 0 # Cumulative Probability
        i = 0

        for idx, (word, prob) in enumerate(unigram_dictionary.items()):
            p += float(math.pow(prob, power)) / norm_val
            while (i < table_size) and (float(i) / table_size < p):
                table[i] = idx
                i += 1

        self.table = table

    def sampling(self, true_idx):
        """
        true_idx : index of target word
        """
        while True:
            indices = np.random.randint(low=0, high=len(self.table), size=self.count)
            if true_idx not in indices:
                break

        return [self.table[i] for i in indices]
