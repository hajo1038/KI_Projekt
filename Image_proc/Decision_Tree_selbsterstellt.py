#ID3-Algorithmus wird hier verwendet. 
#Soll einfacher sein als CART-Algorithmus, da hier kein Gini-Index erechnet werden muss.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def entropy(self, labels):
        # Berechnung der Entropie für die gegebenen Labels
        classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, feature, labels, labels_left, labels_right):
        # Berechnung des Information Gains für das gegebene Feature und für die Labels
        total_entropy = self.entropy(labels)
        weighted_entropy = 0
        # subset_labels = labels[feature == v]
        weight_left = len(labels_left) / len(labels)
        weight_right = len(labels_right) / len(labels)
        entropy_left = self.entropy(labels_left)
        entropy_right = self.entropy(labels_right)
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        information_gain = total_entropy - weighted_entropy
        return information_gain


    def split_data(self, feature, labels, split_value):
        # Aufteilen der Daten basierend auf dem gegebenen Feature und Schwellenwert
        left_indices = np.where(feature <= split_value)
        right_indices = np.where(feature > split_value)
        #left_labels = labels[left_indices]
        #right_labels = labels[right_indices]
        return left_indices, right_indices

    def find_best_split(self, features, labels):
        # Finden des besten Features und Schwellenwerts für die Aufteilung
        best_gain = 0
        best_feature = None
        best_threshold = None
        num_features = features.shape[1]
        for feature_idx in range(num_features):
            feature = features[:, feature_idx]
            unique_values = np.unique(feature)
            for value in unique_values:
                left_ids, right_ids = self.split_data(feature, labels, value)
                gain = self.information_gain(feature, labels, labels[left_ids], labels[right_ids])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = value
        return best_feature, best_threshold

    def build_tree(self, features, labels):
        # Erstellen eines leeren Entscheidungsbaums
        tree = {}
        tree['feature'] = None
        tree['threshold'] = None
        tree['left'] = None
        tree['right'] = None

        # if max_depth

        if len(np.unique(labels)) == 1:
            # Wenn alle Labels gleich sind, haben wir ein Blatt erreicht
            tree['label'] = np.unique(labels)[0]
            return tree

        if features.shape[1] == 0:
            # Wenn keine Features mehr vorhanden sind, geben wir das häufigste Label zurück
            tree['label'] = np.argmax(np.bincount(labels))
            return tree

        best_feature, best_threshold = self.find_best_split(features, labels)

        if best_feature is None:
            # Wenn kein bestes Feature gefunden wurde, haben wir ein Blatt erreicht
            tree['label'] = np.argmax(np.bincount(labels))
            return tree

        left_indices, right_indices = self.split_data(features[:, best_feature], labels, best_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            # Wenn eine Seite der Aufteilung leer ist, geben wir das häufigste Label zurück
            tree['label'] = np.argmax(np.bincount(labels))
            return tree

        tree['feature'] = best_feature
        tree['threshold'] = best_threshold
        tree['left'] = self.build_tree(features[left_indices], labels[left_indices])
        tree['right'] = self.build_tree(features[right_indices], labels[right_indices])
        return tree

    def fit(self, features, labels, max_depth=None):
        # Trainieren des Decision Trees
        self.tree = self.build_tree(features, labels)

    def predict(self, features, labels):
        tp = 0
        for id_x, datapoint in enumerate(features):
            node = self.tree
            while(True):
                splitting_feature = node['feature']
                threshold = node['threshold']
                if datapoint[splitting_feature] <= threshold:
                    node = node["left"]
                    if 'label' in node:
                        prediction = node['label']
                        tp += prediction == labels[id_x]
                        break
                else:
                    node = node["right"]
                    if 'label' in node:
                        prediction = node['label']
                        tp += prediction == labels[id_x]
                        break
        print(f"Accuracy: {tp/len(labels)}")


df = pd.read_csv("../Image_proc/data_with_features_without_white.csv")
df_labels = df["label"]
df_labels = pd.get_dummies(df_labels, columns=["label"])
labels = df_labels.to_numpy()
features = df[["Lines", "Contours Colour", "Contours Size Colour", "Fast"]].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

my_tree = MyDecisionTreeClassifier()
my_tree.fit(X_train, np.argmax(y_train, axis=1), max_depth=4)
my_tree.predict(X_train, np.argmax(y_train, axis=1))
my_tree.predict(X_test, np.argmax(y_test, axis=1))