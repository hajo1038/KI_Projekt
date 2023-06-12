import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


df=pd.read_csv(r'data_with_features_with_FAST.csv')
df.head()

X = df[["Lines", "Contours", "Contours Size", "Keypoints"]] # definiert die ersten 3 Spalten als Feautures
y = df["label"] # definiert die letzte Spalte als Kategorie (entspricht dem Ergebnis)
# trennt die Test- und Trainingsdaten / Übergeben werden die Feautues und die Kategorien / die Anzahl der Testdaten (30% zum Testen
# 70% zum Trainieren, random_state wenn auf None werden die Daten immer unterschiedlich zugeteilt, sonst immer die Gleichen)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


model = DecisionTreeClassifier(random_state=0, max_depth=6) # baut den Entscheidungsbaum mit maximaler Tiefe von 2
# Trainiert den Entscheidungsbaum mit den Trainingsdaten (bestehend aus Input und dem richtigen Ergbenis)
model.fit(X_train, y_train)

print(model.score(X_test, y_test)) # Gibt das Ergebnis des Trainings zurück / wie viele Inputs wurden auf das richtige Ergebnis gemappt

plot_tree(model) # Plottet den Ergebnisbaum