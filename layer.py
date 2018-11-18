import numpy as np
import math
import time
# Skip gram model
# CBOW model
# softmax model
# hierarchical model

def Sigmoid(x):
    # clip input to -10~10
    x = np.clip(x, -10, 10)
    return (1 / 1 + np.exp(-x))

def Skipgram(currentWord, contextWords, inputVectors, outputVectors,
             use_loss='Softmax', tokens=None, negative_sampling = None):
    """
    currentWord    : center word String
    contextWords   : list of string, which has 2 * C words
    inputVectors   : access by row number, such as inputVectors[5]...
    outputVectors  : access by row number same as inputVectors
    use_loss       : Softmax, Hierarchical Softmax, etc...
    tokens         : get index used at inputVectors
    """
    loss = 0.0
    N, V = inputVectors.shape
    C = len(contextWords)

    gradsIn = np.zeros(inputVectors.shape)
    gradsOut = np.zeros(outputVectors.shape)
    hiddenVectors = np.zeros((1, V))
    score = np.zeros((N, 1))

    index = tokens[currentWord]
    hiddenVectors = inputVectors[index] # current word to hidden vector

    if use_loss == 'Softmax':
        # Softmax forwarding
        for i in range(C):
            word = contextWords[i]
            word_idx = tokens[word]
            score[index] += np.dot(hiddenVectors, outputVectors[index].T) # (1, 1)

        exp_out = np.exp(score)
        softmax = exp_out / np.sum(exp_out)
        dsoftmax = softmax.copy()
        # Softmax backwarding
        for i in range(C):
            word = contextWords[i]
            word_idx = tokens[word]
            loss = -np.sum(np.log(softmax[word_idx]))
            dsoftmax[word_idx] -= 1

        loss /= N
        #print (dsoftmax.shape, hiddenVectors.shape)
        gradsOut = np.dot(dsoftmax, hiddenVectors.reshape(1,-1))
        gradshidden = np.dot(dsoftmax.T, outputVectors)
        gradsIn[index] = gradshidden

        # return loss, gradsIn, gradsOut

    #elif use_loss == 'Hierarchical_Softmax':

    elif use_loss == 'Negative_Sampling':
        # get unigram distribution and negative sampling index
        negative_sample_idx = negative_sampling.sampling(index)
        # negative sampling model forwarding
        # negative sampling model backwarding
    return loss, gradsIn, gradsOut

def CBOW(currentWord, contextWords, inputVectors, outputVectors,
             use_loss='Softmax', tokens=None, negative_sampling=None):
    """
    currentWord    : center word String
    contextWords   : list of string, which has 2 * C words
    inputVectors   : access by row number, such as inputVectors[5]...
    outputVectors  : access by row number same as inputVectors
    use_loss       : Softmax, Hierarchical Softmax, etc...
    tokens         : get index used at inputVectors
    """
    loss = 0.0
    N, V = inputVectors.shape
    C = len(contextWords)

    gradsIn = np.zeros(inputVectors.shape)
    gradsOut = np.zeros(outputVectors.shape)
    hiddenVectors = np.zeros((1, V))
    score = np.zeros((N, 1))

    index = tokens[currentWord]

    if use_loss == 'Softmax':
        # Softmax forwarding
        for i in range(C):
            word = contextWords[i]
            word_idx = tokens[word]
            hiddenVectors += inputVectors[word_idx]

        score = np.dot(outputVectors, hiddenVectors.T)
        exp_score = np.exp(score)
        softmax = exp_score / np.sum(exp_score)
        dsoftmax = softmax.copy()

        # Softmax backwarding
        loss = -np.sum(np.log(softmax[index]))
        loss /= N

        dsoftmax[index] -= 1
        gradsOut = np.dot(dsoftmax, hiddenVectors)
        gradshidden = np.dot(dsoftmax.T, outputVectors)

        for i in range(C):
            word = contextWords[i]
            word_idx = tokens[word]
            gradsIn[word_idx] = gradshidden

        # return loss, gradsIn, gradsOut

    #elif use_loss == 'Hierarchical_Softmax':

    elif use_loss == 'Negative_Sampling':
        # get unigram distribution and negative sampling index
        negative_sample_idx = negative_sampling.sampling(index)
        # negative sampling model forwarding
        
        # negative sampling model backwarding
    return loss, gradsIn, gradsOut

# reference by github page
# Unigram Table : https://github.com/deborausujono/word2vecpy/blob/master/word2vec.py
class Unigram_Table:
    #--- 3/4 power of unigram distribution selected by mikolov et al. 2013
    #--- make table and recall with index
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab, count, unigram_dictionary=dict()):
        self.count = count
        vocab_size = len(vocab)
        power = 0.75
        table_size = int(1e8) # unigram table length
        table = np.zeros(table_size, np.uint32)

        norm_val = sum(math.pow(uni_prob, power) for uni_prob in unigram_dictionary.values())

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
