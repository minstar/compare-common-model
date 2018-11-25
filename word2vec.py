import numpy as np
import random


def skipgram(currentWord, contextWords, tokens, inputVectors, outputVectors):
    """
    Skip-gram 모델 구현

    Arguments:
    currentWord -- the center word string
    contextWords -- the context words. 2*C개의 string이 들어있는 list
    tokens -- {key : 단어, value : index 숫자} 의 dictionary
    inputVectors -- 모든 "input" word vectors. 2차원의 numpy array이며 token에 해당하는 row로 접근
                    ex) token 253에 대한 word vector : inputVectors[253]
    outputVectors -- 모든 "output" word vectors. inputVectors와 마찬가지로 접근

    Return:
    cost -- Skip-gram의 cost function 값
    grad -- 두 word vector에 대한 gradient
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE

    #--- get the shape of input matrix, hidden vector and output matrix ---#
    N, V = inputVectors.shape
    C = len(contextWords)
    hiddenVectors = np.zeros((1, V))
    out = np.zeros((N, 1))

    #--- forward skipgram model ---#
    cur_index = tokens[currentWord]             # current word index
    hiddenVectors += inputVectors[cur_index]    # (1, 3)

    for i in range(C):
        word = contextWords[i]
        index = tokens[word]
        out[index] += np.dot(hiddenVectors, outputVectors[index].T) # (1, 1)

    exp_out = np.exp(out)
    softmax = exp_out / np.sum(exp_out)
    dsoftmax = softmax.copy()

    #--- backward of skipgram model and compute the cost (or loss) ---#
    for i in range(C):
        word = contextWords[i]
        index = tokens[word]
        cost = -np.sum(np.log(softmax[index]))
        dsoftmax[index] -= 1

    cost /= N # average the cost
    gradOut = np.dot(dsoftmax, hiddenVectors)   # (5, 3)
    dhidden = np.dot(dsoftmax.T, outputVectors) # (1, 3)
    gradIn[cur_index] = dhidden                 # (1, 3)

    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, contextWords, tokens, inputVectors, outputVectors):
    """
    CBOW 모델 구현

    Arguments:
    currentWord -- the center word string
    contextWords -- the context words. 2*C개의 string이 들어있는 list
    tokens -- {key : 단어, value : index 숫자} 의 dictionary
    inputVectors -- 모든 "input" word vectors. 2차원의 numpy array이며 token에 해당하는 row로 접근
                    ex) token 253에 대한 word vector : inputVectors[253]
    outputVectors -- 모든 "output" word vectors. inputVectors와 마찬가지로 접근

    Skip-gram 모델과 같은 구성
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape) # (5, 3)
    gradOut = np.zeros(outputVectors.shape) # (5, 3)

    ### YOUR CODE HERE
    N,V = inputVectors.shape # (5, 3) for test_word2vec
    C = len(contextWords)
    hiddenVectors = np.zeros((1, V))

    #--- forward of CBOW model ---#
    cur_index = tokens[currentWord]  # "c" : 2

    for i in range(C):
        word = contextWords[i]                   # "a" , "b" ...
        index = tokens[word]                     # "a" : 0, "b" : 1, ...
        hiddenVectors += inputVectors[index]     # (1, 3)

    out = np.dot(outputVectors, hiddenVectors.T) # (5, 1)
    exp_out = np.exp(out)
    softmax = exp_out / np.sum(exp_out)

    #--- compute the cost (or loss) ---#
    cost = -np.sum(np.log(softmax[cur_index]))
    cost /= N                                    # average the cost (or loss)

    #--- backward of CBOW model ---#
    dsoftmax = softmax.copy()
    dsoftmax[cur_index] -= 1
    gradOut = np.dot(dsoftmax, hiddenVectors)    # (5, 3)
    dhidden = np.dot(dsoftmax.T, outputVectors)  # (1, 3)

    for i in range(C):
        word = contextWords[i]
        index = tokens[word]                     # add gate = distribute the gradients
        gradIn[index] = dhidden                  # (1, 3)

    ### END YOUR CODE

    return cost, gradIn, gradOut

##############################################
# 테스트 함수입니다. 절대 수정하지 마세요!!! #
##############################################
def test_word2vec():
    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = np.random.randn(10,3)
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])


    print("=== Results ===")
    print(skipgram("c", ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:]))
    print(skipgram("c", ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:]))
    print(cbow("a", ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:]))
    print(cbow("a", ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:]))


if __name__ == "__main__":
    test_word2vec()
