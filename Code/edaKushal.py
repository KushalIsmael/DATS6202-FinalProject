import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
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
#check % null values in each column
missing = df.isnull().sum()/(len(df))*100
print(missing)
#check columns with over 50% missing values
print(missing.where(missing>50))
#check for total null values in df
print(df.isnull().sum().sum())
#drop non predictive fields
df = df.drop(columns =['state','county','communityname','community','fold',])
#create dataset that drops columns where 50% of values are null
dfdrop = df.dropna(axis=1,thresh=997)
#impute values with mean for full dataset
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(df)
SimpleImputer()
dfimpmean = imp_mean.transform(df)
#impute values with mean for full dataset
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(dfdrop)
SimpleImputer()
dfdropimpmean = imp_mean.transform(dfdrop)

plt.figure(0)
plt.plot(df['ViolentCrimesPerPop'], 'o')
plt.title('Violent Crimes Per 100k')
plt.ylabel('Total number of violent crimes per 100K population')
plt.xlabel('Community index')
plt.savefig('violent.png')
