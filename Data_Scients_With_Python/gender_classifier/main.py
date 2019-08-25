#importamos un arbol de deciones
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160,60,38], [166,65,40], [190,90,47],[175,64,39],
     [177,70,40], [172, 71, 34], [150,40,38], [176,55,42], [199,90,37],[145,34,32]
] 

#Array X va asociado con array Y

Y = ['male', 'male', 'female', 'male', 'female', 'female','male','male','female','male','female','female']

#Creamos el arbol de decisiones
clf = tree.DecisionTreeClassifier()
#Entremaos el arbol con nuestros datos
clf = clf.fit(X,Y)

#clasificador usando random forest
#https://victorzhou.com/blog/intro-to-random-forests/
clforest = ensemble.RandomForestClassifier()
clforest = clforest.fit(X,Y)

#Clasificador usando KNN
#https://www.youtube.com/watch?v=UqYde-LULfs
clfKNN = neighbors.KNeighborsClassifier()
clfKNN = clfKNN.fit(X,Y)

#una vez entrenado el arbol hacemos una prueba de prediccion basado en los datos que le dimos
prediction = clf.predict([[190,70,43]])

#Prediccion usando RandomForest
predictionForest = clforest.predict([[190,70,43]])

#Preddicion usando KNN
predictionKNN = clfKNN.predict([[190,70,43]])


print('Prediccion usando Arbol de Decision {}'.format(prediction))
print('Prediccion usando Random Forest {}'.format(predictionForest))
print('Prediccion usando KNN {}'.format(predictionKNN))