from importlib.resources import path
from os import lseek
import numpy as np
import pandas as pd
import os

# define the location of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv'
# load the dataset
df = pd.read_csv(url, header=None)
# summarize shape
print(df.shape)

# get the current path, path to data
curdir = os.getcwd()
data_path = os.path.join(curdir, "../Datasets/")

# write to Data file
bank_file = os.path.join(data_path, 'banknote.csv')
with open(bank_file, 'w') as csvfile:
    df.to_csv(path_or_buf=csvfile, header=False, index=False)
    
# Normalize the data
norm_bank_file = os.path.join(data_path, 'normalized_banknote.csv')


# write normalized to data file
df_norm = df
for i in range(4):
    df_norm[i] = (df[i] - df[i].mean())/df[i].std()
with open(norm_bank_file, 'w') as csvfile:
    df_norm.to_csv(path_or_buf=csvfile, header=False, index=False)

