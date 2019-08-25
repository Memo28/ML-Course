import numpy as np
from lightfm.datasets import fetch_movielens
#Libreria que tiene algoritmos de recomendacion
from lightfm import LightFM


#Obtener los datos y darles formato
#Obtenemos datos de peliculas que tengan un rating de al menos 4 puntos de 5 
data = fetch_movielens(min_rating=4.0)

#lastfm nos regresa un objecto con varios atributos entre los cuales estan train y test
#si queremos ver que otros atributos trae el objeto lo podemos imprimir
#print(repr(data))

print(repr(data['train']))
print(repr(data['test']))

#creamos nuestro modelo, se le tiene que pasar una funcion de perdida
#en este caso wrap -> 
#loss function mientras mas alto el valor mayor es la acertividad del modelo 
model = LightFM(loss = 'warp')

#Entrenamos el modelo, donde epochs va a ser el numero de veces que se va a entrenar
model.fit(data['train'], epochs = 30, num_threads = 2)


def sample_recomendation(model,data,user_ids):
    n_users, n_items = data['train'].shape

    for user_id in user_ids :
        
        #obtenemos las peliculas que ya le gustaron a ese usuario
        known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #basado en el las que ya le gustan, predecimos y guardamos en un arreglo las que le pueden gustar
        scores = model.predict(user_id, np.arange(n_items))

        #ordenamos de menor a mayor
        top_items = data['item_labels'][np.argsort(-scores)]

        #imprimimos los 3 primeros por usuario
        print ('User {}'.format(user_id))
        print('Basado en lo que te ha gustado')

        for x in known_positive[:3]:
            print("         {}".format(x))

        print('Te recomendamos')

        for x in top_items[:3]:
            print("    {}".format(x))


sample_recomendation(model,data,[12,34,15])



