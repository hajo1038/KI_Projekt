import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import image_processing
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

df = pd.read_csv(r'data_with_features_without_white.csv')
df.head()

X = df[["Contours Colour", "Contours Size Colour", "Lines"]] # definiert die ersten 3 Spalten als Features
y = df["label"] # definiert die letzte Spalte als Kategorie (entspricht dem Ergebnis)
# trennt die Test- und Trainingsdaten / Übergeben werden die Feautues und die Kategorien / die Anzahl der Testdaten (30% zum Testen
# 70% zum Trainieren, random_state wenn auf None werden die Daten immer unterschiedlich zugeteilt, sonst immer die Gleichen)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(random_state=0, max_depth=9) # baut den Entscheidungsbaum mit maximaler Tiefe von 2
# Trainiert den Entscheidungsbaum mit den Trainingsdaten (bestehend aus Input und dem richtigen Ergebnis)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) # Gibt das Ergebnis des Trainings zurück / wie viele Inputs wurden auf das richtige Ergebnis gemappt



net_predictions = model.predict(X_test)
my_net_precisions = precision_score(y_test, net_predictions, average=None)
my_net_recalls = recall_score(y_test, net_predictions, average=None)
my_net_f1s = f1_score(y_test, net_predictions, average=None)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plt.setp(ax, xticks=np.unique(np.argmax(y_test, axis=1)), xlabel="Anzahl der Tafeln")

# create subplots
ax[0].bar(np.unique(np.argmax(y_test, axis=1)), my_net_precisions, color='red')
ax[0].set_title("Precision des neuronalen Netzes")
ax[1].bar(np.unique(np.argmax(y_test, axis=1)), my_net_recalls, color='blue')
ax[1].set_title("Recall des neuronalen Netzes")
ax[2].bar(np.unique(np.argmax(y_test, axis=1)), my_net_f1s, color='green')
ax[2].set_title("F1-Score des neuronalen Netzes")
plt.show()


#correct_value = 9
#a = list(image_processing.main2())
#print(model.score([a], [correct_value]))