from Model import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import tensorflow as tf
# from keras.datasets import mnist
from matplotlib import pyplot
from glob import glob
import cv2
import numpy as np
import os
import itertools
import pickle
import json
import csv


CSV_PATH = "./models/grid_search_without_white.csv"

def get_mean_error(pred, y_true):
    diff = np.argmax(y_true, axis=1) - np.argmax(pred, axis=1)
    mean_diff = np.mean(np.abs(diff[np.nonzero(diff)]))
    return mean_diff

def model_configs():
    eta = [0.01]
    batch_size = [16, 32, 64]
    epochs = [300, 500, 800]
    layers = [1, 2, 3]
    nodes = [32, 64, 128]

    configs = list(itertools.product(batch_size, eta, epochs, layers, nodes))
    return configs

def k_fold_grid_search(X, y):
    configs = model_configs()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["batch_size", "eta", "epochs", "layers", "nodes", "accuracy"])
    for config in configs:
        tp = 0
        accuracies = []
        n_batch_size, n_eta, n_epochs, n_layers, n_nodes = config
        for i, (train_index, test_index) in enumerate(skf.split(X, np.argmax(y, axis=1))):
            tp = 0
            train_mean = np.mean(X[train_index])
            train_std = np.std(X[train_index])
            X_train = (X[train_index] - train_mean) / train_std
            X_test = (X[test_index] - train_mean) / train_std
            y_train = y[train_index]
            y_test = y[test_index]
            model = Model()
            model.add_layer(X_train.shape[1])
            for layer_id, layer in enumerate(range(n_layers)):
                if layer_id == 0:
                    model.add_layer(n_nodes)
                else:
                    model.add_layer(int(n_nodes/(layer_id*2)))
            model.add_layer(y_train.shape[1])
            model.train(X_train, y_train, epochs=n_epochs, eta=n_eta, mini_batch_size=n_batch_size)
            model.predict(X_test, y_test)
            y_true = np.argmax(y_test, axis=1)
            predictions = np.argmax(model.predictions, axis=1)
            tp = np.sum(predictions == y_true)
            accuracies.append(tp/y_test.shape[0])
        accuracy = np.mean(accuracies)
        print(accuracy)
        with open(CSV_PATH, "a", newline="") as file:
            writer = csv.writer(file)
            # Write the values row by row
            write_list = list(config)
            write_list.append(accuracy)
            writer.writerow(write_list)


def load_iris():
    df = pd.read_csv("/Users/jonathanhaller/Documents/Studium/Master/Verfahren_der_KI/datasets/iris_data.csv")
    species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df["species"] = df['species'].map(species_to_int)
    df_label = df["species"]
    df_label = pd.get_dummies(df_label)
    df_X = df.drop("species", axis=1)
    X = df_X.to_numpy()
    y = df_label.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def load_ritter_sport_xy(path, feature_names=None, test_size=0.2):
    df = pd.read_csv(path)
    df_labels = df["label"]
    df_files = df["file"]
    df_labels = pd.get_dummies(df_labels, columns=["label"])
    labels = df_labels.to_numpy()
    df.drop(columns=["label", "file"], inplace=True)
    if feature_names == None:
        features = df.to_numpy()
    else:
        features = df[feature_names].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def load_ritter_sport_data():
    images_list = glob(r"../data/*.png")
    image_names = [os.path.basename(file) for file in glob('../data/*.png')]
    features = np.ndarray((len(images_list), 512*512))
    for count, image in enumerate(images_list):
        im = cv2.imread(image)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        features[count, :] = gray_image.flatten()
    # normalizing the image values to have a mean of 0 and std of 1
    features = (features - np.mean(features)) / np.std(features)
    df_labels = pd.read_csv("../Labeling_App/Labels.csv")
    df_labels = pd.get_dummies(df_labels, columns=["label"])
    labels = np.array([df_labels[df_labels["file"] == image].iloc[:, 1:] for image in image_names])
    labels = labels.reshape((labels.shape[0], 15))
    [X_train, X_test, y_train, y_test] = train_test_split(features, labels, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def load_mnist():
    # loading
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # one-hot encoding
    y_train.reshape(-1)
    y_train = np.eye(10)[y_train]
    y_test.reshape(-1)
    y_test = np.eye(10)[y_test]

    X_flat = []
    for i in X_test:
        X_flat.append(i.flatten())
    X_test = X_flat

    X_flat = []
    for i in X_train:
        X_flat.append(i.flatten())
    X_train = X_flat
    # shape of dataset
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return X_train, X_test, y_train, y_test


def main():
    # X_train, X_test, y_train, y_test = load_iris()

    # X_train, X_test, y_train, y_test = load_ritter_sport_data()

    # X_train, X_test, y_train, y_test = load_mnist()

    X_train, X_test, y_train, y_test = load_ritter_sport_xy("../Image_proc/data_with_features_without_white.csv", feature_names=["Lines", "Contours Colour", "Contours Size Colour"])


    # k_fold_grid_search(X_train, y_train)


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    model = Model()
    model.add_layer(X_train.shape[1])
    model.add_layer(128)
    model.add_layer(64)
    model.add_layer(32)
    #model.add_layer(64)
    #model.add_layer(64)
    #model.add_layer(32)
    model.add_layer(y_train.shape[1])
    #model.train(X_train, y_train, epochs=1600, eta=0.01, mini_batch_size=32, val_x=X_val, val_y=y_val)
    model.train(X_train, y_train, epochs=800, eta=0.01, mini_batch_size=16, val_x=X_val, val_y=y_val)

    model.predict(X_test, y_test)

    with open('my_neural_net.pkl', 'wb') as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)

    mean_diff = get_mean_error(model.predictions, y_test)
    print(f"Mean-Error: {mean_diff}")

    y_test_max = np.argmax(y_test, axis=1)
    pred_max = np.argmax(model.predictions, axis=1)
    ConfusionMatrixDisplay.from_predictions(y_test_max, pred_max)
    plt.show()
    plt.plot(model.loss)
    plt.plot(model.val_loss)
    plt.show()


if __name__ == '__main__':
    main()
