import os
import numpy as np
import pandas as pd

from numpy.random import Generator, PCG64

partycolors = {'SVP':'#006400','SP':'#FF0000','FDP':'#4169E1','CVP':'#FFA500',"Grüne":'#008000','BDP':'#800020','glp':'#32CD32'}
partynames  = {v:k for k,v in zip(partycolors.keys(), partycolors.values())}
partyclasses = {k:i for i, (k,v) in enumerate(zip(partycolors.keys(), partycolors.values()))}

def sparsen(row, p, generator=None):
    if generator is None:
        generator = Generator(PCG64(seed=0))
    row = row.copy()
    idx = generator.choice(row.index, round(len(row)*p), replace=False)
    row.loc[idx] = np.nan
    return row

def load_data(folder_name, data_name):

    folder_path = f"../../data/{folder_name}"
    data_path = f"../../data/{folder_name}/{data_name}"

    test_users = pd.read_csv(f'{folder_path}/test_candidates.csv', index_col=0)
    users = pd.read_csv(f'{folder_path}/train_candidates.csv', index_col=0)
    test_reactions = load_all('test_reactions', data_path)
    reactions = load_all('train_reactions', data_path)

    return (users, reactions), (test_users, test_reactions)

def load_all(name, directory):
    tmp = {}
    
    for file in os.listdir(directory):
        if file.startswith(f'{name}_') and file.endswith('.csv'):
            path = os.path.join(directory, file)
            tmp[int(file.split('_')[-1].split('.')[0])] = pd.read_csv(path, index_col=0)
    
    return tmp

def impute_dataframe(dataframe, mean=None):
    if mean is None:
        mean = dataframe.mean()
    return (dataframe - mean).fillna(0) + mean