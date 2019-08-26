from tpot import TPOTClassifier
from sklearn.model_selection  import train_test_split
import pandas as pd
import numpy as np


#Cargamos los datos
telescope = pd.read_csv('MAGIC Gamma Telescope Data.csv')

#Limpiamos y reordenamos los datos
telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop = True)

tele['Class'] = tele['Class'].map({'g' : 0, 'h':1})
tele_class = tele['Class'].values

#Dividimos los datos en entrenamiento, testing y validacion
training_indices, validation_indices = training_indices, testing_indices = train_test_split(
    tele.index, stratify=tele_class, train_size=0.75, test_size=0.25)

#Ocupamos tpot para que nos diga que algoritmo es mejor para nuestro problema
tpot = TPOTClassifier(verbosity = 2)

tpot.fit(tele.drop('Class', axis=1).loc[training_indices].values,
    tele.loc[training_indices, 'Class'].values)

#Calculamos la asertividad 
tpot.score(tele.drop('Class',axis=1).loc[validation_indices].values,
    tele_loc[validation_indices, 'Class'].values)

tpot.export('pipeline.py')



