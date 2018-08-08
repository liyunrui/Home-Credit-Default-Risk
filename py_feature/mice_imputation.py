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


#---------------------------------------------
# Step2: imputatiing based on imputation method u chossed
#---------------------------------------------

def imputation():
    '''
    visit_sequence: order in which we visit the columns, monotone" (default), "roman", "arabic", "revmonotone
    impute_type: probablistic moment matching or posterior predictive distribution (default).
    init_fill_method: "mean" (default), "median", or "random".
    '''
    #-------------------
    # preprocessing for algorithm
    #-------------------
    for drop_targtet in [False]:
        if drop_targtet == True:
            # avoidnig using TARGET to impute, 
            name = 'wo_target'
            X_missing = df.copy()
            X_missing.drop(['TARGET'], axis =1, inplace = True)

        else:
            name = 'w_target'
            X_missing = df.copy()
            # what I expected, maybe the TARGET variable is the auxiliary variable, so w target may be helful for imputation, then local CV is upper.
        
        logging.info('case: {}'.format(name)) 
        logging.info('visit_sequence: {}'.format('monotone')) 
        logging.info('impute_type: {}'.format('col')) 
        logging.info('init_fill_method: {}'.format('mean')) 
        #-------------------
        # core algorithm: input should be array
        #-------------------
        X_filled = MICE(visit_sequence = 'monotone', 
                        impute_type = 'col',
                        init_fill_method = 'mean').complete(X_missing.values)

        #-------------------
        # output
        #-------------------
        X_filled = pd.DataFrame(X_filled, columns = X_missing.columns)

        gc.collect()
        X_filled.SK_ID_CURR = X_filled.SK_ID_CURR.astype(int)
        logging.info('there will be no bugging in merge' if X_filled.SK_ID_CURR.nunique() == df.SK_ID_CURR.nunique() else "opps")

        if drop_targtet == True: 
            pass
        else:
            X_filled.drop(['TARGET'], axis = 1, inplace = True)

        X_filled = pd.merge(X_filled, copy_for_the_following_merge, on = 'SK_ID_CURR', how = 'left')
        logging.info('final_shape : {}'.format(X_filled.shape))

        #-------------------
        # save
        #-------------------
        output_path = '../features/filled_by_mice'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        X_filled.to_hdf(
            os.path.join(output_path, 'base_featurs_filled_mice_{}.h5'.format(name)), 
            'base_featurs_filled_mice_{}.h5'.format(name))

##################################################
# Main
##################################################

s = time.time()

imputation()

e = time.time()
logging.info('{} secs'.format(e-s))




