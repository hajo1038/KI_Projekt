import pandas as pd
from sklearn.model_selection import train_test_split
from My_Decision_Tree import MyDecisionTreeClassifier
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("../Image_proc/data_with_features_without_white.csv")
    df_labels = df["label"]
    df_labels = pd.get_dummies(df_labels, columns=["label"])
    labels = df_labels.to_numpy()
    features = df[["Lines", "Contours Colour", "Contours Size Colour", "Fast"]].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

    my_tree = MyDecisionTreeClassifier()
    my_tree.fit(X_train, np.argmax(y_train, axis=1), max_depth=5)
    # my_tree.predict(X_train, np.argmax(y_train, axis=1))
    my_tree.predict(X_test, np.argmax(y_test, axis=1))
    # ConfusionMatrixDisplay.from_predictions(np.argmax(y_test, axis=1), my_tree.predictions)
    # plt.show()