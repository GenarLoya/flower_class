from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# cargando los datos

flowers_class = pd.read_csv('src/iris.csv', engine="python")

# obteniendo los datos del dataset
print(flowers_class)

# variable predictoras
X = flowers_class.iloc[:, 0:3]

# variable a predecir
Y = flowers_class.iloc[:, 4]

print("VARIABLES PREDICTORAS:")
print(X)
print("VARIABLES A PREDECIR:")
print(Y)

print("-----TRAINING------")
# 80 & 20
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.80, random_state=0)
print("X TRAIN:")
print(X_train.info())

# Classifier
clf = DecisionTreeClassifier()

# Training
tree_flowers = clf.fit(X_train, Y_train)

# Graph
print("-----GRAPH------")
fig = plt.figure(figsize=(25, 20))
tree.plot_tree(tree_flowers, feature_names=list(
    X.columns.values), class_names=list(Y.values), filled=True)
plt.show()

Y_pred = tree_flowers.predict(X_test)

print("PREDICTIONS Y:")
print(Y_pred)


matriz_confusion = confusion_matrix(Y_test, Y_pred)
# matriz_confusion

print("-----MATRIZ DE CONFUSION------")

print("Verdaderos Negativos:")
verdaderos_negativos = matriz_confusion[0][0]
print(verdaderos_negativos)
print("Falsos Negativos:")
falsos_negativos = matriz_confusion[1][0]
print(falsos_negativos)
print("Falsos Positivos:")
falsos_positivos = matriz_confusion[0][1]
print(falsos_positivos)
print("Verdaderos Positivos:")
verdaderos_positivos = matriz_confusion[1][1]
print(verdaderos_positivos)
print("-----PRECISION CALCULATION------")
print("Precisión:")
precision = np.sum(matriz_confusion.diagonal())/np.sum(matriz_confusion)
print(precision)
print("Precisión No:")
precision_no = verdaderos_negativos / sum(matriz_confusion[0,])
print(precision_no)
print("Precisión Si:")
precision_si = verdaderos_positivos / sum(matriz_confusion[1,])
print(precision_si)
