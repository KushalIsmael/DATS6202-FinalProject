import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputerimport seaborn as sns

# Functions
def findCorrelations(correlations, cutoff=0.9):
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
# Make plots
plt.figure(0)
missing[missing > 0].plot.barh()
plt.ylabel('Feature')
plt.xlabel('Percent')
# plt.xticks(rotation=80)
plt.savefig('missing.png', dpi=300, bbox_inches='tight')

plt.figure(1)
plt.hist(df['ViolentCrimesPerPop'])
plt.xlabel('Number of Violent Crimes Per 100k Population')
plt.title('Histogram of Violent Crimes')
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')

#subset with race fields
dfrace = df[['racepctblack','racePctWhite','racePctAsian','racePctHisp','ViolentCrimesPerPop']]
#pair plot of race and violent crimes
plt.title('Pair Plot of Race and Violent Crimes')
sns.pairplot(dfrace)
plt.show()

#correlation matrix of % race and violent crimes
corrace = dfrace.corr()
sns.heatmap(corrace, annot=True)
plt.show()

#subset with age fields
dfage = df[['agePct12t21','agePct12t29','agePct16t24','agePct65up','ViolentCrimesPerPop']]
#pair plot for age and violent crimes
sns.pairplot(dfage)
plt.title('Pair Plot of Age and Violent Crimes')
plt.show()

#correlation matrix for age and violent crimes
corrage = dfage.corr()
sns.heatmap(corrage, annot=True)
plt.show()


#check columns with over 50% missing values
print(missing.where(missing>50))
#check for total null values in df
print(df.isnull().sum().sum())
#drop non predictive fields
df = df.drop(columns =['state','county','communityname','community','fold',])
#create dataset that drops columns where 50% of values are null

plt.figure(2)
plt.plot(df['ViolentCrimesPerPop'], 'o')
plt.title('Violent Crimes Per 100k')
plt.ylabel('Total number of violent crimes per 100K population')
plt.xlabel('Community index')
plt.savefig('violent.png')

# Impute missing values

X = df.drop('ViolentCrimesPerPop', axis=1)
y = df['ViolentCrimesPerPop']

# imp = IterativeImputer(max_iter=10, random_state=0)
# Ximpute = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Highly correlated features
to_remove = findCorrelations(X_train.corr())
X_train = X_train.drop(X_train.columns[to_remove], axis=1)
# Drop police columns with NA values
X_train = X_train.dropna(axis=1)

piped = Pipeline([('mlp', MLPRegressor(random_state=1))])
max_neurons = 100
max_layers = 5
params = []
for i in np.arange(2, max_layers + 1, 1):
    for j in np.arange(5, max_neurons + 1, 5):
        out = tuple(np.repeat(j + 1, i + 1))
        params.append(out)

param_grid = [{'mlp__activation': ['logistic', 'tanh', 'relu'],
               'mlp__hidden_layer_sizes': params}]

gs = GridSearchCV(estimator=piped,
                  param_grid=param_grid,
                  scoring='neg_mean_squared_error',
                  verbose=2,
                  cv=5)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
