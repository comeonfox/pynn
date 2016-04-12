#coding:utf8

import numpy as np
import sys

class Layer(object):
    """Layer of NN. Not output layer.
    Attributes:
        input (Layer): A Layer object, if it's None, then this is a Input Layer.
        a (np.ndarray): outputs after activiation, shape (size,).
        size (np.uint32): number of nodes in the layer.
        weight (np.ndarray): shape (input.size, size).
        activiation (func): func(x) --> y.
    """

