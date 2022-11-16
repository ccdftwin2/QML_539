from importlib.resources import path
import numpy as np
import pandas as pd
import os

# get the current path, path to data
curdir = os.getcwd()
data_path = os.path.join(curdir, "../Datasets/")
# get the file of the heart
heart_file = os.path.join(data_path, 'heart.csv')

# load the dataset
df = pd.read_csv(heart_file)
# summarize shape
print(df.shape)

# Get the names
names = list(df)

# get the normalized heart file
norm_heart_file = os.path.join(data_path, 'normalized_heart.csv')

# write normalized to data file
df_norm = df
for i in names:
    if i != 'output':
        df_norm[i] = (df[i] - df[i].mean())/df[i].std()
with open(norm_heart_file, 'w') as csvfile:
    df_norm.to_csv(path_or_buf=csvfile, header=True, index=False)

