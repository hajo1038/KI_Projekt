import math
from Layer import Layer
import numpy as np


def softmax(vector):
    vector -= np.max(vector)
    return np.exp(vector) / np.sum(np.exp(vector))


def softmax_grad(vector):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    # https://medium.com/intuitionmath/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    softmax_output = softmax(vector)
    s = softmax_output.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def relu(vector):
    return vector * (vector > 0)


def relu_d(vector):
    return (vector > 0).astype(int)


def binary_cross_entropy(y_pred_proba, y_true):
    return (-1)*((y_true * math.log2(y_pred_proba)) + ((1-y_true) * math.log2(1-y_pred_proba)))


def multi_cross_entropy(y_pred_proba, y_true):
    ce_sum = 0
    for i in range(len(y_pred_proba)):
        ce_sum += y_true[i] * math.log2(y_pred_proba[i])
    return ce_sum * (-1)


def multi_cross_entropy_d(pred_proba, y_true):
    # derivate with respect to the predicted probabilities
    return -(y_true/pred_proba)


class Model:
    def __init__(self):
        self.layers = []
        self.loss = []

    def add_layer(self, units_in_layer):
        # assert isinstance(layer, Layer)
        layer = Layer(units_in_layer)
        self.layers.append(layer)
        if len(self.layers) > 1:
            layer.connect_layer(self.layers[-2])

    def train(self, x, y, epochs):
        # x has dimensions: datapoints x features, y has dimensions: datapoints x Labels
        if x.shape[1] != self.layers[0].units:
            raise Exception("Input dimension does not match dimension of first layer")
        elif y.shape[1] != self.layers[-1].units:
            raise Exception("Output dimension does not match dimension of last layer")
        # go through every datapoint (one row of feature values)
        for epoch in range(epochs):
            print(f"Epoche {epoch + 1} von {epochs}")
            loss = 0
            for id_x, datapoint in enumerate(x):
                for id_l, layer in enumerate(self.layers):
                    if id_l == 0:
                        self.layers[id_l].values = datapoint
                        self.layers[id_l].activations = datapoint
                        continue
                    # calculate the values of the neurons based on the weights and values of the previous layer
                    self.layers[id_l].values = np.dot(layer.weights, self.layers[id_l - 1].activations) + layer.biases
                    # ReLU activation function in all other up to the last one
                    if id_l < (len(self.layers) - 1):
                        self.layers[id_l].activations = relu(self.layers[id_l].values)
                    # use softmax in the last layer
                    else:
                        self.layers[id_l].activations = softmax(self.layers[id_l].values)
                if y.shape[1] > 2:
                    loss = multi_cross_entropy(self.layers[-1].activations, y[id_x, :])
                elif y.shape[1] == 2:
                    loss = binary_cross_entropy(self.layers[-1].activations, y[id_x, :])
                self.update(datapoint, y[id_x, :], 0.001)
            print(f"{loss}\n")
            self.loss.append(loss)

    def update(self, features, labels, eta):
        nabla_b = [np.zeros(self.layers[id_l].biases.shape) for id_l in range(1, len(self.layers))]
        nabla_w = [np.zeros(self.layers[id_l].weights.shape) for id_l in range(1, len(self.layers))]
        # for x, y in zip(features, labels):
        #     delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        #     nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #     nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        delta_nabla_b, delta_nabla_w = self.backprop(features, labels)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        for id_l in range(1, len(self.layers)):
            self.layers[id_l].weights = self.layers[id_l].weights - (eta * nabla_w[id_l-1])
            self.layers[id_l].biases = self.layers[id_l].biases - (eta * nabla_b[id_l-1])

        # self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        # TODO Alle Dimensionen nochmal überprüfen!
        nabla_b = [np.zeros(self.layers[id_l].biases.shape) for id_l in range(1, len(self.layers))]
        nabla_w = [np.zeros(self.layers[id_l].weights.shape) for id_l in range(1, len(self.layers))]
        # feedforward
        # activation = x
        # activations = [x]  # list to store all the activations, layer by layer
        # zs = []  # list to store all the z vectors, layer by layer
        # backward pass
        delta = np.dot(multi_cross_entropy_d(self.layers[-1].activations, y)[:, np.newaxis].T, softmax_grad(self.layers[-1].values))
        nabla_b[-1] = delta.T[:, 0]
        nabla_w[-1] = np.outer(delta, self.layers[-2].activations)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, len(self.layers)):
            z = self.layers[-l].values
            sp = relu_d(z)
            delta = (self.layers[-l + 1].weights.T @ nabla_b[-l + 1]) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, self.layers[-l - 1].activations.transpose())
        return nabla_b, nabla_w

# TODO Feed-Forward in Funktion auslagern

    def predict(self, x, y):
        tp = 0
        for id_x, datapoint in enumerate(x):
            for id_l, layer in enumerate(self.layers):
                if id_l == 0:
                    self.layers[id_l].values = datapoint
                    self.layers[id_l].activations = datapoint
                    continue
                # calculate the values of the neurons based on the weights and values of the previous layer
                self.layers[id_l].values = np.dot(layer.weights, self.layers[id_l - 1].activations) + layer.biases
                # ReLU activation function in all other up to the last one
                if id_l < (len(self.layers) - 1):
                    self.layers[id_l].activations = relu(self.layers[id_l].values)
                # use softmax in the last layer
                else:
                    self.layers[id_l].activations = softmax(self.layers[id_l].values)
            if y.shape[1] > 2:
                loss = multi_cross_entropy(self.layers[-1].activations, y[id_x, :])
                prediction = np.argmax(self.layers[-1].activations)
                y_true = np.argmax(y[id_x, :])
                tp += int(np.array_equal(prediction, y_true))
            elif y.shape[1] == 2:
                loss = binary_cross_entropy(self.layers[-1].activations, y[id_x, :])
        print(f"Accuracy = {tp/y.shape[0]}\n")
