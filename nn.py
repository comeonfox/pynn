#coding:utf8
import math
import numpy as np
import sys

def softmax(w):
    """softmax.
    :param w: np.array
    """
    return np.exp(w) / np.sum(np.exp(w))

def cross_entropy(N, y, y_hat):
    """cross entropy function, used as the cost function.
    :param N: number of samples in the batch.
    :param y: matrix of N by output_dim.
    :param y_hat: predictions, same shape with y.
    """
    # * means Hadamard product here, because both y and y_hat have
    # type of numpy.ndarry
    x = y * np.log(y_hat)
    return -1.0 * np.sum(x) / N

class NN(object):
    """A basic neural network."""
    def __init__(self, input, hidden, output, batch_size, num_passes, epsilon):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param batch_size: batch_size
        """
        self.input = input
        self.hidden = hidden
        self.output = output
        self.batch_size = batch_size
        self.num_passes = num_passes
        self.epsilon = epsilon

        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)

        # store predictions and labels for a batch
        self.y = np.zeros((self.batch_size, self.output))

        # store nodes activations
        self.ao = np.zeros((self.batch_size, self.output))
        self.ah = np.zeros((self.batch_size, self.hidden))
        self.ai = np.zeros((self.batch_size, self.input))

        # store biases
        self.bi = np.zeros((self.batch_size, self.hidden))
        self.bh = np.zeros((self.batch_size, self.output))


    def feedForward(self, inputs=None):
        """This takes only 1 sample a time."""
        # input activations
        self.ai = inputs

        # hidden activations
        self.ah = np.tanh(np.dot(self.ai, self.wi)) # + self.bi)

        # output activations
        self.ao = softmax(np.dot(self.ah, self.wo)) # + self.bh)

    def backPropagate(self):
        """ This is what happens in 1 batch.
        :param targets: y values
        :param epsilon: learning rate
        :return: update weights and current error
        """
        delta3 = self.ao - self.y
        delta2 = delta3.dot(self.wo.T) * (1 - np.power(self.ah, 2))

        # get gradients
        # TODO: add regularization to w1 and w2.
        dw2 = np.dot(self.ah.T, delta3)
        dw1 = np.dot(self.ai.T, delta2)
        db1 = delta2
        db2 = delta3

        # update params
        self.wi -= self.epsilon * dw1
        self.wo -= self.epsilon * dw2
        #self.bi -= self.epsilon * db1
        #self.bh -= self.epsilon * db2

    def feed(self, Y, X):
        """A generator yields a matrix of shape (batch_size, input_dim)."""
        self.Y = Y
        self.X = X

    def train(self):
        for _ in xrange(self.num_passes):
            pos = 0
            for __ in xrange(int(math.ceil(len(self.X) / self.batch_size))):
                self.y = self.Y[pos:self.batch_size + pos]
                inputs = self.X[pos:self.batch_size + pos, :]
                self.feedForward(inputs)
                self.backPropagate()
                print cross_entropy(self.batch_size, self.y, self.ao)
            #print self.ao
        #print self.bi[:10, :]
        #print "self.wi: >>>", self.wi.shape
        #print self.wi
        #print "self.wo: >>>", self.wo.shape
        #print self.wo

    def predict(self, y, testset):
        right = 0
        self.feedForward(testset)
        predictions = np.argmax(self.ao, axis=1)
        diff = zip(predictions, y)
        prd = open('prd.out', 'w')
        for p, yy in diff:
            print >> prd, p, yy
            if p == yy:
                right += 1
        prd.close()
        print "precision: %.7f" %  (1.0 * right / len(y))

