#coding:utf8

import sys
from nn import NN
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

def classify_digits():
    digits = load_digits()
    images_and_labels = list(zip(digits.images, digits.target))
    n_samples = len(digits.target)
    # X
    data = digits.images.reshape(n_samples, -1)
    # Y
    Y = np.zeros((n_samples, 10))
    for i, y in enumerate(Y):
        Y[i, digits.target[i]] = 1
#    for index, (image, label) in enumerate(images_and_labels[:4]):
#        plt.subplot(2, 4, index + 1)
#        plt.axis('off')
#        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#        plt.title('Training: %i' % label)
#        plt.show()
    nn = NN(64, 30, 10, 1, 100, 0.002)
    nn.feed(Y, data)
    nn.train()
    nn.predict(digits.target, data)


if __name__ == '__main__':
    classify_digits()
