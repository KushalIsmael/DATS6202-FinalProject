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

    #--KNR--
    knr = neighbors.KNeighborsRegressor(n_neighbors=3)
    knr.fit(X_train,y_train)
    y_pred = knr.predict(X_test)
    knrscore = knr.score(y_test,y_pred)
    knrmse = mean_squared_error(y_test, y_pred)

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

    evaltable = pd.DataFrame({
        'Model':['SVR','KNR','DTR','RFR','MLP'], # add any other
        'Accuracy Score': [svrscore,knrscore,dtrscore,rfrscore],
        'MSE':[svrmse,knrmse,dtrmse,rfrmse]
    })



    return evaltable.style.background_gradient(cmap='Blues')