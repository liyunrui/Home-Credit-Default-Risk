'''

@author: Ray

output may need to be modified to include SK_ID_CURR for merging.

Reference: https://github.com/iskandr/fancyimpute
'''
from fancyimpute import MICE # for imputing
import pandas as pd
import numpy as np
import os
import gc
import time
import sys
sys.path.append('../py_model')
from utils import init_logging
import logging 

#---------------------------------------------
# Step1: preprocessing for MICE
#---------------------------------------------
# setting

log_dir = '../log_mice_inputation'
init_logging(log_dir)

df = pd.read_hdf('../features/base_featurs.h5','base_featurs')

copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()

no_need_to_inpute = ['index']
df.drop(no_need_to_inpute, axis = 1, inplace = True)
# handling with infinity
df.replace([np.inf, -np.inf], np.nan, inplace = True)

logging.info('shape of base_featurs : {}'.format(df.shape) )

# computing the threshold
th = df[~df.TARGET.isnull()].TARGET.mean()
# loading prediction result
test = pd.read_csv('../py_model/sub.csv')
test['TARGET'] = test.TARGET.apply( lambda x: 1 if x >= th else 0)
# new TARGET field
TARGET = df[~df.TARGET.isnull()].TARGET.tolist() + test.TARGET.tolist()
# change
df['TARGET'] = TARGET

del TARGET, test, th
gc.collect()

#---------
# setting
#---------
log_dir = '../log_mice_inputation'
init_logging(log_dir)

from fancyimpute import MICE # for imputing
output_filled_ls = []
# unit test if MICE algorithm will change the order of DataFrame
for i in range(2):
    if i == 0:
        X_missing = df[df.TARGET == 0]
        #-------------------
        # core algorithm: input should be array
        #-------------------
        logging.info('visit_sequence: {}'.format('monotone')) 
        logging.info('impute_type: {}'.format('col')) 
        logging.info('init_fill_method: {}'.format('mean')) 
        logging.info('target == 0')
        X_filled = MICE(visit_sequence = 'monotone', 
                        impute_type = 'col',
                         init_fill_method = 'mean').complete(X_missing.values)
    elif i == 1:
        X_missing = df[df.TARGET == 1]
        #-------------------
        # core algorithm: input should be array
        #-------------------
        logging.info('visit_sequence: {}'.format('monotone')) 
        logging.info('impute_type: {}'.format('col')) 
        logging.info('init_fill_method: {}'.format('mean')) 
        logging.iffo('target == 1')
        X_filled = MICE(visit_sequence = 'monotone', 
                        impute_type = 'col',
                         init_fill_method = 'mean').complete(X_missing.values)
    else:
        pass
       
    
    output_filled_ls.append(pd.DataFrame(X_filled, columns = X_missing.columns))

#---------
# output
#---------
import gc
# merge
X_filled = pd.concat(output_filled_ls, axis = 1)
# drop fake TARGET
X_filled.drop(['TARGET'], axis = 1, inplace = True)
gc.collect()
# unit test
X_filled.SK_ID_CURR = X_filled.SK_ID_CURR.astype(int)
logging.info('there will be no bugging in merge' if X_filled.SK_ID_CURR.nunique() == df.SK_ID_CURR.nunique() else "opps")
# merge original TARGET back
X_filled = pd.merge(X_filled, copy_for_the_following_merge, on = 'SK_ID_CURR', how = 'left')
logging.info('final_shape : {}'.format(X_filled.shape))

#-------------------
# save
#-------------------
output_path = '../features/filled_by_mice'
if not os.path.isdir(output_path):
    os.mkdir(output_path)

X_filled.to_hdf(
    os.path.join(output_path, 'base_featurs_filled_mice_supervised.h5'), 'base_featurs_filled_mice_supervised')

