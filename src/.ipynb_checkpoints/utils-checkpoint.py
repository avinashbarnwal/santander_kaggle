
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pickle
from sklearn.model_selection import KFold
from time import time
from datetime import datetime
import gc
from itertools import chain


# In[ ]:


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)


# In[ ]:


def to_pickles(df, path, split_size=3, inplace=True):
    
    """
    path = '../output/mydf'
    
    write '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)
    
    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(f'{path}/{i:03d}.pkl')
    return


def read_pickles(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.                      format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def save_obj(obj, name):
    with open('../data/output/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name ):
    with open('../data/output/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)   

def result_append(Name,CV_LOSS,param,LB,number_of_folds,sampling):
    path = '../data/output/'
    try:
        file_name         = path + "result.json"
        result            = pd.read_json(file_name)
        row,col           = result.shape
        result.loc[row,:] = [CV_LOSS,Name,sampling,param,number_of_folds]
        result.to_json("../data/output/result.json")
        
    except:    
        result            = pd.DataFrame(columns=['CV_LOSS','Name','sampling','params','number_of_folds'])
        result.loc[0,:]   = [CV_LOSS,Name,sampling,param,number_of_folds]
        result.to_json("../data/output/result.json")