from train_util import *
from layer import *
from preprocessing import *

import numpy as np
import time
import math

class Word2Vec():

    def __init__(self, vocab_size=100000, embedding_size=300, context_size=5, neg_samples=5, optimize='sgd',
                architecture='cbow', use_loss='Negative_Sampling', learning_rate=1e-3, epochs=1, decay_rate = 0.99,
                decay_step = 100000, verbose=100, sample_table=None, encoder=None):
        self.params = {}
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.neg_samples = neg_samples
        self.optimize = optimize
        self.architecture = architecture
        self.use_loss = 'Negative_Sampling'
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.verbose = verbose
        self.sample_table = sample_table
        self.encoder = encoder

        self.params['inputVectors'] = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_size)) # (100000, 300)
        self.params['outputVectors'] = np.random.randn(self.vocab_size, self.embedding_size) / self.embedding_size

    def gradient_update(self, grads, grads_idx):
        for key, grad in grads.items():
            idx = grads_idx[key]
            if 'in' in key:
                self.params['inputVectors'][idx] -= self.learning_rate * grad
            elif 'out' in key:
                self.params['outputVectors'][idx] -= self.learning_rate * grad

    def loss(self, currentWord, contextWords):
        inputVectors = self.params['inputVectors']
        outputVectors = self.params['outputVectors']

        if self.architecture == 'cbow':
            loss, grads, grads_idx = CBOW(currentWord, contextWords, inputVectors, outputVectors,
                                                self.use_loss, self.sample_table, self.encoder)
        elif self.architecture == 'skipgram':
            loss, grads, grads_idx = Skipgram(currentWord, contextWords, inputVectors, outputVectors,
                                                self.use_loss, self.sample_table, self.encoder)

        return loss, grads, grads_idx

    def train(self, data):
        loss = 0.0
        sum_loss = 0.0

        for epoch in range(self.epochs):
            start = time.time()
            for data_idx in range(len(data)):
                # current word and context words indexes
                currentWord, contextWords = generate_batch(data, data_idx, self.context_size)

                if self.architecture == 'cbow':
                    loss, grads, grads_idx = self.loss(currentWord, contextWords)
                    self.gradient_update(grads, grads_idx)

                elif self.architecture == 'skipgram':
                    for contextWord in contextWords:
                        loss, grads, grads_idx = self.loss(currentWord, contextWords)
                        self.gradient_update(grads, grads_idx)

                sum_loss += loss

                if (data_idx+1) % self.decay_step == 0:
                    self.learning_rate *= self.decay_rate

                if (data_idx+1) % self.verbose == 0:
                    one_time = time.time()
                    print ('idx: %d, loss: %f, time: %f, ' % (data_idx+1, sum_loss/self.verbose, one_time-start))
                    sum_loss = 0.0
                    start = time.time() # restart


def test_word2vec():
    data = np.concatenate((np.random.randint(0,5,100),np.random.randint(5,10,100)))
    vocab_size = 10
    embedding_size = 6
    context_size = 2
    epoch = 10

    freq = {}
    unigram_distribution = {}

    for word in data:
        if word not in freq:
            freq[word] = 1
        freq[word] += 1

    for word, cnt in freq.items():
        unigram_distribution[word] = cnt / len(data)

    sample_table = Unigram_Table(vocab_size, 2, unigram_distribution)
    huffman_encode = Huffman_encoding(unigram_distribution)
    huffman_encode.encoding()
    node_lst, node_dict, reverse_node_dict = huffman_encode.make_node_dict()

    architectures = ['cbow', 'skipgram']
    use_loss = ['Softmax', 'Negative_Sampling', 'Hierarchical_Softmax']
    learning_rate = {'Softmax':1e-3, 'Negative_Sampling':1e-3, 'Hierarchical_Softmax':1e-3}

    for architecture in architectures:
        for loss_type in use_loss:
            print ('Testing Model :%s, loss : %s' %(architecture, loss_type))
            model = Word2Vec(vocab_size, embedding_size, context_size, architecture=architecture, verbose=len(data),
                            use_loss=loss_type, learning_rate = learning_rate[loss_type], sample_table=sample_table, encoder=node_dict)
            model.train(data)
