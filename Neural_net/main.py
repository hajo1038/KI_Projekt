from Model import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
# from keras.datasets import mnist
from matplotlib import pyplot
from glob import glob
import cv2
import numpy as np
import os


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

    X_train, X_test, y_train, y_test = load_ritter_sport_data()

    # X_train, X_test, y_train, y_test = load_mnist()

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    model = Model()
    model.add_layer(X_train.shape[1])
    model.add_layer(20)
    model.add_layer(10)
    model.add_layer(y_train.shape[1])
    model.train(X_train, y_train, epochs=50, eta=0.01, mini_batch_size=10, val_x=None, val_y=None)

    model.predict(X_test, y_test)
    # print(f"Pred: {model.predictions} True: {y_test}")
    plt.plot(model.loss)
    plt.plot(model.val_loss)
    plt.show()


if __name__ == '__main__':
    main()
