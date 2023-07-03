import numpy as np


class MyDecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.depth = 0
        self.predictions = []

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
        # left_labels = labels[left_indices]
        # right_labels = labels[right_indices]
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

    def build_tree(self, features, labels, max_depth=None):
        # Erstellen eines leeren Entscheidungsbaums
        tree = {}
        tree['feature'] = None
        tree['threshold'] = None
        tree['left'] = None
        tree['right'] = None
        tree['depth'] = 1

        if max_depth is not None:
            if self.depth > max_depth:
                tree['label'] = np.argmax(np.bincount(labels))
                self.depth -= 1
                return tree

        if len(np.unique(labels)) == 1:
            # Wenn alle Labels gleich sind, haben wir ein Blatt erreicht
            tree['label'] = np.unique(labels)[0]
            self.depth -= 1
            return tree

        if features.shape[1] == 0:
            # Wenn keine Features mehr vorhanden sind, geben wir das häufigste Label zurück
            tree['label'] = np.argmax(np.bincount(labels))
            self.depth -= 1
            return tree

        best_feature, best_threshold = self.find_best_split(features, labels)

        if best_feature is None:
            # Wenn kein bestes Feature gefunden wurde, haben wir ein Blatt erreicht
            tree['label'] = np.argmax(np.bincount(labels))
            self.depth -= 1
            return tree

        left_indices, right_indices = self.split_data(features[:, best_feature], labels, best_threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            # Wenn eine Seite der Aufteilung leer ist, geben wir das häufigste Label zurück
            tree['label'] = np.argmax(np.bincount(labels))
            self.depth -= 1
            return tree

        self.depth += 1

        tree['feature'] = best_feature
        tree['threshold'] = best_threshold
        tree['left'] = self.build_tree(features[left_indices], labels[left_indices], max_depth)
        tree['right'] = self.build_tree(features[right_indices], labels[right_indices], max_depth)
        return tree

    def fit(self, features, labels, max_depth=None):
        # Trainieren des Decision Trees
        self.tree = self.build_tree(features, labels, max_depth)

    def predict(self, features, labels):
        tp = 0
        self.predictions = []
        for id_x, datapoint in enumerate(features):
            node = self.tree
            while True:
                splitting_feature = node['feature']
                threshold = node['threshold']
                if datapoint[splitting_feature] <= threshold:
                    node = node["left"]
                    if 'label' in node:
                        prediction = node['label']
                        self.predictions.append(prediction)
                        tp += prediction == labels[id_x]
                        break
                else:
                    node = node["right"]
                    if 'label' in node:
                        prediction = node['label']
                        self.predictions.append(prediction)
                        tp += prediction == labels[id_x]
                        break
        print(f"Accuracy: {tp/len(labels)}")
