from math import sqrt
import numpy as np
from numpy.random import randn
from Neuron import Neuron

class Layer:
    def __init__(self, units):
        self.units = units
        self.bias = 0
        self.neurons = []
        self.weights = 0
        self.values = np.zeros(units)
        self.activations = np.zeros(units)

    def connect_layer(self, previous_layer):
        weight_per_neuron = previous_layer.units
        # weight matrix is neurons x neurons_in_previous_layer
        self.weights = np.zeros((self.units, weight_per_neuron))
        for row in self.weights:
            self.weights[row, :] = [sqrt(2.0 / weight_per_neuron) * randn(1)[0] for weight in range(weight_per_neuron)]
        #for unit in range(self.units):


            #neuron = Neuron(weight_per_neuron)
            #self.neurons.append(neuron)
