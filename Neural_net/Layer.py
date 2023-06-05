from math import sqrt
import numpy as np
from numpy.random import randn


class Layer:
    def __init__(self, units):
        self.units = units
        self.biases = np.zeros(units)
        self.neurons = []
        self.weights = 0
        self.nabla_w = 0
        self.nabla_b = np.zeros(units)
        self.values = np.zeros(units)
        self.activations = np.zeros(units)

    def connect_layer(self, previous_layer):
        weight_per_neuron = previous_layer.units
        # weight matrix is neurons x neurons_in_previous_layer
        self.weights = np.zeros((self.units, weight_per_neuron))
        self.nabla_w = np.zeros((self.units, weight_per_neuron))
        for row_count, row in enumerate(self.weights):
            self.weights[row_count, :] = [sqrt(2.0 / weight_per_neuron) * randn(1)[0]
                                          for weight in range(weight_per_neuron)]
