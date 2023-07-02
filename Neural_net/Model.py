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
    epsilon = 1e-13
    # index = [np.argmax(y_true)]
    # if y_pred_proba[index] == 0:
    #    return (-1) * math.log2(y_pred_proba[index] + epsilon)
    # return (-1) * math.log2(y_pred_proba[index])
    for i in range(len(y_pred_proba)):
        ce_sum += y_true[i] * math.log2(y_pred_proba[i] + epsilon)
    return ce_sum * (-1)


def multi_cross_entropy_d(pred_proba, y_true):
    # derivate with respect to the predicted probabilities
    epsilon = 1e-13
    #return pred_proba - y_true
    return -(y_true / (pred_proba+epsilon))


class Model:
    def __init__(self):
        self.layers = []
        self.loss = []
        self.loss_sum = 0
        self.val_loss = []
        self.predictions = []
        self.true_positives = 0
        self.accuracies = []
        self.val_accuracies = []
        self.dropout = 0.0

    def add_layer(self, units_in_layer):
        # assert isinstance(layer, Layer)
        layer = Layer(units_in_layer)
        self.layers.append(layer)
        if len(self.layers) > 1:
            layer.connect_layer(self.layers[-2])

    def train(self, x, y,  epochs, eta=0.01, mini_batch_size=1, dropout=0.0, val_x=None, val_y=None):
        # x has dimensions: datapoints x features, y has dimensions: datapoints x Labels
        self.dropout = dropout
        if x.shape[1] != self.layers[0].units:
            raise Exception("Input dimension does not match dimension of first layer")
        elif y.shape[1] != self.layers[-1].units:
            raise Exception("Output dimension does not match dimension of last layer")

        x_batches = [x[k:k + mini_batch_size] for k in range(0, x.shape[0], mini_batch_size)]
        y_batches = [y[k:k + mini_batch_size] for k in range(0, y.shape[0], mini_batch_size)]

        # go through every datapoint (one row of feature values)
        for epoch in range(epochs):
            self.loss_sum = 0
            self.true_positives = 0
            update_count = 0
            print(f"Epoche {epoch + 1} von {epochs}")
            # if there is validation data, calculate the validation loss
            if (val_x is not None) and (val_y is not None):
                self.validation(val_x, val_y)

            for x_batch, y_batch in zip(x_batches, y_batches):
                update_count += 1
                self.update(x_batch, y_batch, eta)
            print(f"Update-Count: {update_count}\n")
            self.loss.append(self.loss_sum/x.shape[0])
            self.accuracies.append(self.true_positives / y.shape[0])
            print(f"Train-Loss: {self.loss[-1]}")
            print(f"Train-Acc: {self.accuracies[-1]}")
            if (val_x is not None) and (val_y is not None):
                print(f"Val-Loss: {self.val_loss[-1]}")
                print(f"Val-Acc: {self.val_accuracies[-1]}\n")

    def update(self, features, labels, eta):
        nabla_b = [np.zeros(self.layers[id_l].biases.shape) for id_l in range(1, len(self.layers))]
        nabla_w = [np.zeros(self.layers[id_l].weights.shape) for id_l in range(1, len(self.layers))]
        for x, y in zip(features, labels):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        for id_l in range(1, len(self.layers)):
            self.layers[id_l].weights = self.layers[id_l].weights - (eta * nabla_w[id_l - 1]/features.shape[0])
            self.layers[id_l].biases = self.layers[id_l].biases - (eta * nabla_b[id_l - 1]/features.shape[0])
        # self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(self.layers[id_l].biases.shape) for id_l in range(1, len(self.layers))]
        nabla_w = [np.zeros(self.layers[id_l].weights.shape) for id_l in range(1, len(self.layers))]
        # feedforward
        self.forward_pass(x, y)

        delta = np.dot(multi_cross_entropy_d(self.layers[-1].activations, y)[:, np.newaxis].T, softmax_grad(self.layers[-1].values))
        nabla_b[-1] = delta.T[:, 0]
        nabla_w[-1] = np.outer(delta, self.layers[-2].activations)

        for l in range(2, len(self.layers)):
            z = self.layers[-l].values
            sp = relu_d(z)
            delta = (self.layers[-l + 1].weights.T @ nabla_b[-l + 1]) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, self.layers[-l - 1].activations.transpose())
        return nabla_b, nabla_w

    def forward_pass(self, x, y):
        for id_l, layer in enumerate(self.layers):
            if id_l == 0:
                self.layers[id_l].values = x
                self.layers[id_l].activations = x
                continue
            elif self.dropout != 0.0:
                dropout_matrix = np.random.rand(self.layers[id_l - 1].activations.shape[0]) < (1-self.dropout)
                self.layers[id_l - 1].activations = self.layers[id_l - 1].activations * dropout_matrix
                self.layers[id_l - 1].activations = self.layers[id_l - 1].activations / (1-self.dropout)
            # calculate the values of the neurons based on the weights and values of the previous layer
            self.layers[id_l].values = np.dot(layer.weights, self.layers[id_l - 1].activations) + layer.biases
            # ReLU activation function in all other up to the last one
            if id_l < (len(self.layers) - 1):
                self.layers[id_l].activations = relu(self.layers[id_l].values)
            # use softmax in the last layer
            else:
                self.layers[id_l].activations = softmax(self.layers[id_l].values)

        loss = multi_cross_entropy(self.layers[-1].activations, y)
        prediction = np.argmax(self.layers[-1].activations)
        y_true = np.argmax(y)
        true_positive = np.array_equal(prediction, y_true)

        self.loss_sum += loss
        self.true_positives += int(true_positive)

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
                self.predictions.append(self.layers[-1].activations.tolist())
            elif y.shape[1] == 2:
                loss = binary_cross_entropy(self.layers[-1].activations, y[id_x, :])
        print(f"Accuracy = {tp/y.shape[0]}\n")

    def validation(self, val_x, val_y):
        val_loss = 0
        val_loss_sum = 0
        true_positives = 0
        for id_x, datapoint in enumerate(val_x):
            for id_l, layer in enumerate(self.layers):
                if id_l == 0:
                    self.layers[id_l].values = datapoint
                    self.layers[id_l].activations = datapoint
                    continue
                # calculate the values of the neurons based on the weights and values of the previous layer
                self.layers[id_l].values = np.dot(layer.weights,
                                                  self.layers[id_l - 1].activations) + layer.biases
                # ReLU activation function in all other up to the last one
                if id_l < (len(self.layers) - 1):
                    self.layers[id_l].activations = relu(self.layers[id_l].values)
                # use softmax in the last layer
                else:
                    self.layers[id_l].activations = softmax(self.layers[id_l].values)
            if val_y.shape[1] > 2:
                val_loss = multi_cross_entropy(self.layers[-1].activations, val_y[id_x, :])
                val_loss_sum += val_loss
                prediction = np.argmax(self.layers[-1].activations)
                y_true = np.argmax(val_y[id_x, :])
                true_positives += int(np.array_equal(prediction, y_true))
            elif val_y.shape[1] == 2:
                val_loss = binary_cross_entropy(self.layers[-1].activations, val_y[id_x, :])

        self.val_loss.append(val_loss_sum / val_x.shape[0])
        self.val_accuracies.append(true_positives / val_y.shape[0])
