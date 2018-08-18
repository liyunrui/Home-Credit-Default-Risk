import pandas as pd
import numpy as np
import time
import os
import multiprocessing as mp # for speeding up some process
import sys
sys.path.append('../py_model')
from utils import init_logging
import logging 
import gc

# loading data
df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
print (df.shape)
copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()
no_need_to_inpute = ['index']
df.drop(no_need_to_inpute, axis = 1, inplace = True)
# handling with infinity
df.replace([np.inf, -np.inf], np.nan, inplace = True)
print (df.shape)

# computing the threshold
th = df[~df.TARGET.isnull()].TARGET.mean()
# loading prediction result
test = pd.read_csv('../py_model/sub.csv')
test['TARGET'] = test.TARGET.apply( lambda x: 1 if x >= th else 0)

# test from df
df[df.TARGET.isnull()][['SK_ID_CURR','TARGET']].SK_ID_CURR.tolist() == test.SK_ID_CURR.tolist()

# new TARGET field
TARGET = df[df.TARGET.notnull()].TARGET.tolist() + test.TARGET.tolist()
# check
print (len(df[df.TARGET == 0]) + len(df[df.TARGET == 1]) == len(df))

#---------
# setting
#---------
log_dir = '../log_mice_inputation'
init_logging(log_dir)
X_missing = df[df.TARGET == 1]
X_missing.drop(['TARGET'], axis = 1)
#-------------------
# core algorithm: input should be array
#-------------------
from fancyimpute import MICE # for imputing

logging.info('visit_sequence: {}'.format('monotone')) 
logging.info('impute_type: {}'.format('col')) 
logging.info('init_fill_method: {}'.format('mean')) 
logging.info('target == 1')
X_filled1 = MICE(visit_sequence = 'monotone', 
                impute_type = 'col',
                 init_fill_method = 'mean').complete(X_missing.values)
    