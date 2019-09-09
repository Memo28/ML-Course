import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import pandas as pd
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split
from matplotlib import style

#Getting the data
df = quandl.get("WIKI/AMZN")
print(df.head())

df = df[['Adj. Close']]

#Days to predict
forecast_out = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)


#Create an space for x predicted values
X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)

#Create an space for y predicted values
y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)

#Split the data 80% train 20% test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#array to store the predicted scores

#Trained with Linear Regression
# https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86
lr  = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print('Lr confidence', lr_confidence)


#Trained with Deciscion Tree Regressor
#https://www.saedsayad.com/decision_tree_reg.htm
lg = DecisionTreeRegressor(random_state=0)
lg.fit(x_train, y_train)
lg_confidence = lg.score(x_test, y_test)
print('Lg confidence', lg_confidence)



br = BayesianRidge(compute_score=True)
br.fit(x_train, y_train)
br_confidence = br.score(x_test, y_test)
print('GP', br_confidence)

#Getting the best result
score = [
    {   
        'Method' : 'Linear Regression',
        'Score' : lr_confidence
    },
    {
        'Method' : 'Decision Tree Regression',
        'Score' : lg_confidence
    },
    {
        'Method' : 'Bayesian Ridge',
        'Score' : br_confidence
    }
]

dd = pd.DataFrame.from_dict(score)
max_obj = dd.iloc[dd['Score'].idxmax(),:].to_dict()
print(max_obj)


#Making prediction with each model
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)


lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

lg_prediction = lg.predict(x_forecast)
print(lg_prediction)

br_prediction = br.predict(x_forecast)
print(br_prediction)


#Ploting the data
fig, ax = plt.subplots()

ax.set_title('Comparacion de modelos')
ax.set_ylabel('Adj Close')

ax.plot(x_forecast, label="data de validacion")
ax.plot(lr_prediction, label="Linear Regression")
ax.plot(lg_prediction, label="Logistic Regression")
ax.plot(br_prediction, label="Bayesian Ridge")
plt.legend()
plt.show()

