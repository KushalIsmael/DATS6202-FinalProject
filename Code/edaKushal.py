import pandas as pd
import numpy as np
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
#impute values with mean


