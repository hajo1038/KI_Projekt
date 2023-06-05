from Model import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("/Users/jonathanhaller/Documents/Studium/Master/Verfahren_der_KI/datasets/iris_data.csv")
    species_to_int = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df["species"] = df['species'].map(species_to_int)
    df_label = df["species"]
    df_label = pd.get_dummies(df_label)
    df_X = df.drop("species", axis=1)
    X = df_X.to_numpy()
    y = df_label.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.3, random_state=42)

    model = Model()
    model.add_layer(X_train.shape[1])
    model.add_layer(8)
    model.add_layer(y_train.shape[1])
    model.train(X_train, y_train, epochs=100)

    model.predict(X_test, y_test)
    plt.plot(model.loss)
    plt.show()


if __name__ == '__main__':
    main()
