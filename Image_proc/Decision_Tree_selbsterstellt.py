#ID3-Algorithmus wird hier verwendet. 
#Soll einfacher sein als CART-Algorithmus, da hier kein Gini-Index erechnet werden muss.

import numpy as np

class MyDecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def entropy(self, labels):
        # Berechnung der Entropie für die gegebenen Labels
        classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    #def information_gain(self, feature, labels):
        # Berechnung des Information Gains für das gegebene Feature und für die Labels
        #total_entropy = self.entropy(labels)
        #unique_values = np.unique(feature)
        #weighted_entropy = 0
        #for v in unique_values:
            #subset_labels = labels[feature == v]
            #weight = len(subset_labels) / len(labels)
            #weighted_entropy += weight * self.entropy(subset_labels)
        #information_gain = total_entropy - weighted_entropy
        #return information_gain

    # Berechnung des Information Gains für das gegebene Feature und Labels
        total_entropy = self.entropy(labels)
        unique_values = np.unique(feature)
        weighted_entropy = 0
        for v in unique_values:
            subset_labels = labels[feature == v]
            weight = len(subset_labels) / len(labels)
            subset_entropy = self.entropy(subset_labels)
            weighted_entropy += weight * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain


    def split_data(self, feature, labels, split_value):
        # Aufteilen der Daten basierend auf dem gegebenen Feature und Schwellenwert
        left_indices = np.where(feature == split_value)
        right_indices = np.where(feature != split_value)
        left_labels = labels[left_indices]
        right_labels = labels[right_indices]
        return left_labels, right_labels

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
                left_labels, right_labels = self.split_data(feature, labels, value)
                gain = self.information_gain(feature, labels)
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

    
    
    #def build_tree(self, features, labels):
        # Rekursive Funktion zum Aufbau des Entscheidungsbaums
        #if len(np.unique(labels)) == 1:
            # Wenn alle Labels gleich sind, haben wir ein Blatt erreicht
            #return np.unique(labels)[0]
        #best_feature, best_threshold = self.find_best_split(features, labels)
        #if best_feature is None:
            # Wenn kein bestes Feature gefunden wurde, haben wir ein Blatt erreicht
            #return np.argmax(np.bincount(labels))
        #left_indices, right_indices = self.split_data(features[:, best_feature], labels, best_threshold)
        #tree = {}
        #tree['feature'] = best_feature
        #tree['threshold'] = best_threshold
        #tree['left'] = self.build_tree(features[left_indices], labels[left_indices])
        #tree['right'] = self.build_tree(features[right_indices], labels[right_indices])
        #return tree

    def fit(self, features, labels):
        # Trainieren des Decision Trees
        self.tree = self.build_tree(features, labels)

    #def predict_single(self, example, tree):
        # Vorhersage für ein einzelnes Beispiel basierend auf dem gegebenen Entscheidungsbaum
        #if isinstance(tree, int):
            # Wenn es sich um eine Blattknoten handelt, geben wir das Label zurück
            #return tree
        #feature = tree['feature']
        #threshold = tree['threshold']
        #if example[feature] == threshold:

