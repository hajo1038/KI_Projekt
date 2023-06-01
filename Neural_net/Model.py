import math
from Layer import Layer
import numpy as np


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def relu(vector):
    return vector * (vector > 0)


def binary_cross_entropy(y_pred_proba, y_true):
    return (-1)*((y_true * math.log2(y_pred_proba)) + ((1-y_true) * math.log2(1-y_pred_proba)))


def multi_cross_entropy(y_pred_proba, y_true):
    ce_sum = 0
    for i in range(len(y_pred_proba)):
        ce_sum += y_true[i] * math.log2(y_pred_proba)
    return ce_sum * (-1)


class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, units_in_layer):
        # assert isinstance(layer, Layer)
        layer = Layer(units_in_layer)
        self.layers.append(layer)
        if len(self.layers) > 1:
            layer.connect_layer(self.layers[-2])

    def train(self, x, y):
        # x has dimensions: datapoints x features, y has dimensions: datapoints x Labels
        if x.shape[1] != self.layers[0].units:
            raise Exception("Input dimension does not match dimension of first layer")
        elif y.shape[1] != self.layers[-1].units:
            raise Exception("Output dimension does not match dimension of last layer")
        # go through every datapoint (one row of feature values)
        for id_x, datapoint in enumerate(x):
            for id_l, layer in enumerate(self.layers):
                if id_l == 0:
                    self.layers[id_l].values = layer.weights * datapoint + layer.bias
                    continue
                # calculate the values of the neurons based on the weights and values of the previous layer
                self.layers[id_l].values = layer.weights * self.layers[id_l - 1].activations + layer.bias
                # ReLU activation function in all other up to the last one
                if id_l < (len(self.layers) - 1):
                    self.layers[id_l].activations = relu(self.layers[id_l].values)
                # use softmax in the last layer
                else:
                    self.layers[id_l].activations = softmax(self.layers[id_l].values)
            if y.shape[1] > 2:
                #
                loss = multi_cross_entropy(self.layers[-1].activations, y[id_x, :])
            elif y.shape[1] == 2:
                loss = binary_cross_entropy(self.layers[-1].activations, y[id_x, :])
