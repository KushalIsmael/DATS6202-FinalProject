import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, neighbors, neural_network
from sklearn.neural_network import MLPRegressor
#todo create train and split here for input into models

#https://scikit-learn.org/stable/modules/svm.html#regression
svmr = svm.SVR()
svmr.fit(#xtrain,#ytrain)
SVR()
svmr.predict(#ytest)
#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
knnr = neighbors.KNeighborsRegressor(n_neighbors=3)
knnr.fit(#xtrain,#ytrain)
KNeighborsRegressor(...)
knnr.predict(#ytest)

#https://scikit-learn.org/stable/modules/tree.html#regression
dtr = tree.DecisionTreeRegressor()
dtr = dtr.fit(#Xtrain, #ytrain)
dtr.predict(#ytest)

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
mlp = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
mlp.predict(X_test[:2])
array([-0.9..., -7.1...])
>>> regr.score(X_test, y_test)