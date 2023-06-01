from math import sqrt

import numpy as np
from numpy.random import randn

class Neuron:
    def __init__(self, weights):
        w = [sqrt(2.0/weights)*randn(1)[0] for weight in range(weights)]
        self.weights = np.asarray(w)
        self.value = 0