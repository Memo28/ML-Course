import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


dates = []
prices = []


#Leemos el archivo y llenamos los arreglos
#el formato del archivo es descargado de https://finance.yahoo.com
#Y tiene Date, Open, High, Low, Close, Adj Close, Volume
def get_data(filename):
    with  open(filename) as csvfile:
        csvFileReader = csv.reader(csvfile)
        #saltamos la primera file porque son los nombres
        next(csvFileReader)

        for row in csvFileReader:
            #formateamos la fecha y la guardamos en el arreglo de fechas
            dates.append(int(row[0].split('-')[0]))
            #formateamos el precio y lo guardamos en el arreglo de precios
            prices.append(float(row[1]))
    
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1))

    #Ocuapos un SVR para hacer regresion y obtene la prediccion del siguiente valor
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3)
    svr_rbf =SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_lin.fit(dates,prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates,prices)

    plt.scatter(dates,color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates),color='red', label='RBF model')
    plt.plot(dates, svr_poly, color='blue', label = 'Polynomial model')
    plt.plot(dates, svr_lin, color='green', label = 'Linear model')

    print('OK')

    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    #Regresamos la preddicon del dia siguiente
    return svr_rbf.predictx(x)[0],svr_poly.predict(x)[0], svr_lin.predict(x)[0]


get_data('AAPL.csv')


print(dates,prices)
predicted_price = predict_prices(dates,prices,26)

print(predicted_price)


