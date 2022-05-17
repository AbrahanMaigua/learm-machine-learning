# alboles de decicion
"""
https://scikit-learn.org/stable/modules/tree.html#tree-classification
Exiten 2 tipos de arbolde de decicion:

los alboles de clasificacion,
los albole de regrecion.

son um metodo de apredizaje suérvisado no 
parametrado se utilizan para clasificar y 
la regrecion el objetivo es crear un modelo 
que prediga el valor fr una variable de
distisno  mediante el apredizaje de reglas 
de decicio simples  deduciendas 
de los caraterísticas de los datos.
"""
import pandas as pd
from sklearn import tree


#importanten los datos debes completos
# sin niguna fila vacia
#datos 
x = [[0, 0], [1, 1]]
y = [0, 1]

# entrenamiento del modelo
clf = tree.DecisionTreeClassifier()
pred = clf.fit(x, y)

print(clf.predict([[2.,2.]]))
print(clf.predict([[1.,1.]]))


# usando el conjunto de ire que trae la bibliteca
# sklearn.datasets
# https://en.wikipedia.org/wiki/Iris_flower_data_set#:~:text=The%20Iris%20flower%20data%20set,example%20of%20linear%20discriminant%20analysis.
# pdemos crear un albol de decicion y previsualizarlo

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()
x, y = iris.data, iris.target
clf  = tree.DecisionTreeClassifier()
iris_model = clf.fit(x, y)

t = tree.plot_tree(iris_model)
plt.show()


"""
# importaciones nesesarias
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# limpiamos y genramos los datos
iris = load_iris()
X = iris.data
y = iris.target

# gwnweamos los datos de entenamientos
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# creamos el modelo
clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
clf.fit(X_train, y_train)

"""