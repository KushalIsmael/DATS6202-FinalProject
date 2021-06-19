import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors, neural_network, ensemble
from sklearn.metrics import mean_squared_error


def modeleval(X,y,test):
    """
    Run multiple ML models with X and y data and return table to evaluate

    :param X: predictor variables
    :param y: target variable
    :param test: float of amount to test

    :return: table of model with corresponding R**2 and MSE
    """

    #--train test split--
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test)

    #---SVR---
    svr = svm.SVR()
    svr.fit(X_train,y_train)
    y_pred = svr.predict(X_test)
    svrscore = svr.score(y_test,y_pred)
    svrmse = mean_squared_error(y_test,y_pred)

    #--KNR3--
    knr3 = neighbors.KNeighborsRegressor(n_neighbors=3)
    knr3.fit(X_train,y_train)
    y_pred = knr3.predict(X_test)
    knr3score = knr3.score(y_test,y_pred)
    knr3mse = mean_squared_error(y_test, y_pred)

    #--KNR5--
    knr5 = neighbors.KNeighborsRegressor(n_neighbors=5)
    knr5.fit(X_train,y_train)
    y_pred = knr5.predict(X_test)
    knr5score = knr5.score(y_test,y_pred)
    knr5mse = mean_squared_error(y_test, y_pred)

    #--DTR--
    dtr = tree.DecisionTreeRegressor()
    dtr.fit(X_train,y_train)
    y_pred = dtr.predict(X_test)
    dtrscore = dtr.score(y_test,y_pred)
    dtrmse = mean_squared_error(y_test,y_pred)

    #--RFR--
    rfr = ensemble.RandomForestRegressor()
    rfr.fit(X_train,y_train)
    y_pred = rfr.predict(X_test)
    rfrscore = rfr.score(y_test,y_pred)
    rfrmse = mean_squared_error(y_test,y_pred)

    #--table of results--
    evaltable = pd.DataFrame({
        'Model':['SVR','KNR3','KNR5','DTR','RFR','MLP'],
        'Accuracy Score': [svrscore, knr3score, knr5score, dtrscore, rfrscore],
        'MSE':[svrmse,knr3mse, knr5mse, dtrmse, rfrmse]
    })

    return evaltable.style.background_gradient(cmap='Blues')


plt.fig(0)
plt.hist(y,bins=20, label='y')
plt.hist(y_pred,bins=20,label='y_pred')