import pandas as pd
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
# Check first 10 rows
print(df.head(10))
# Check info of dataset
print(df.info())


#todo check for nulls
#todo decide to drop field with nulls or replace with calc value

