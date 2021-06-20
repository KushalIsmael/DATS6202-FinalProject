import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import svm

model = pd.read_csv('modeleval-full.csv')

best_model = model.loc[model['Rank']==1].reset_index()
paramsdict = best_model['Parameters'].to_dict()
params = eval(paramsdict[0])

def findCorrelations(correlations, cutoff=0.9):
    """

    :param correlations:
    :param cutoff:
    :return:
    """
    corr_mat = abs(correlations)
    varnum = corr_mat.shape[1]
    original_order = np.arange(0, varnum + 1, 1)
    tmp = corr_mat.copy(deep=True)
    np.fill_diagonal(tmp.values, np.nan)
    maxAbsCorOrder = tmp.apply(np.nanmean, axis=1)
    maxAbsCorOrder = (-maxAbsCorOrder).argsort().values
    corr_mat = corr_mat.iloc[list(maxAbsCorOrder), list(maxAbsCorOrder)]
    newOrder = original_order[list(maxAbsCorOrder)]
    del (tmp)
    deletecol = np.repeat(False, varnum)
    x2 = corr_mat.copy(deep=True)
    np.fill_diagonal(x2.values, np.nan)
    for i in range(varnum):
        if not (x2[x2.notnull()] > 0.9).any().any():
            print('No correlations above threshold')
            break
        if deletecol[i]:
            continue
        for j in np.arange(i + 1, varnum, 1):
            if (not deletecol[i] and not deletecol[j]):
                if (corr_mat.iloc[i, j] > cutoff):
                    mn1 = np.nanmean(x2.iloc[i,])
                    mn2 = np.nanmean(x2.drop(labels=x2.index[j], axis=0).values)
                    if (mn1 > mn2):
                        deletecol[i] = True
                        x2.iloc[i, :] = np.nan
                        x2.iloc[:, i] = np.nan
                    else:
                        deletecol[j] = True
                        x2.iloc[j, :] = np.nan
                        x2.iloc[:, j] = np.nan
    newOrder = [i for i, x in enumerate(deletecol) if x]
    return(newOrder)

# Declare an empty list to store each line
lines = []
# Open communities.names for reading text data.
with open ('communities.names', 'rt') as attributes:
    for line in attributes:
        #split each line and keep 2nd element
        lines.append(line.split()[1])

# Read in communities data
df = pd.read_csv('communities.data',header=None)
# Add column names
df.columns = lines
#check size of dataframe
print(df.shape)
# Check first 10 rows
print(df.head(10))
# Check info of dataset
print(df.info(verbose=True))
# replace ? values with numpy nan
df = df.replace('?',np.nan)
# Find the number of columns with missing values
print(df.isnull().any().sum(axis=0))
#check % null values in each column
missing = df.isnull().sum()/(len(df))*100
print(missing)
#check columns with over 50% missing values
print(missing.where(missing>50))
#check for total null values in df
print(df.isnull().sum().sum())
#drop non predictive fields
df = df.drop(columns =['state','county','communityname','community','fold',])

X = df.drop('ViolentCrimesPerPop', axis=1)
y = df['ViolentCrimesPerPop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Highly correlated features
to_remove = findCorrelations(X_train.corr())
X_train = X_train.drop(X_train.columns[to_remove], axis=1)
X_test = X_test.drop(X_test.columns[to_remove], axis=1)

# Drop police columns with NA values
X_train = X_train.dropna(axis=1)
X_test = X_test.dropna(axis=1)
X_test = X_test.drop(columns=['OtherPerCap'])

print(X_train.shape, X_test.shape)

if best_model.iloc[0][1] == 'MLP':
    mlp = MLPRegressor(hidden_layer_sizes=params['mlp__hidden_layer_sizes'], activation=params['mlp__activation'])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

elif best_model['Model'] == 'RFR':
    rfr = RandomForestRegressor(bootstrap=params['bootstrap'], max_depth=params['max_depth'], max_features='sqrt', n_estimators=params['n_estimators'])
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

elif best_model['Model'] == 'DTR':
    dtr = DecisionTreeRegressor(max_depth=params['max_depth'],max_features=params['max_features'])
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)

else :
    svmr = SVR(C=params['C'], epsilon=params['epsilon'], kernel=params['kernel'])
    svmr.fit(X_train, y_train)
    y_pred = svmr.predict(X_test)

plt.figure(0)
plt.title('Violent Crimes per Population Comparison')
plt.hist(y_test,bins=20,label='Actual',alpha=0.5)
plt.hist(y_pred, bins=20, label='Predicted',alpha=0.5)
plt.legend()
plt.savefig('Histogram Comparison', dpi=300, bbox_inches='tight')





