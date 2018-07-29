'''

@author: Ray

output may need to be modified to include SK_ID_CURR for merging.

Reference: https://github.com/iskandr/fancyimpute
'''
from fancyimpute import MICE # for imputing
import pandas as pd
import numpy as np
import multiprocessing as mp # for speeding up some process
import os
import gc
import time
#---------------------------------------------
# Step1: preprocessing for MICE
#---------------------------------------------

df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
no_need_to_inpute = ['SK_ID_CURR','index']
df.drop(no_need_to_inpute, axis = 1, inplace = True)
# handling with infinity
df.replace([np.inf, -np.inf], np.nan, inplace = True)
print ('The shape of DataFrame needed to complete : ',df.shape)


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
    for drop_targtet in [True, False]:
        if drop_targtet == True:
            # avoidnig using TARGET to impute, 
            name = 'wo_target'
            X_missing = df.copy()
            X_missing.drop(['TARGET'], axis =1, inplace = True)

        else:
            name = 'w_target'
            X_missing = df.copy()
            # what I expected, maybe the TARGET variable is the auxiliary variable, so w target may be helful for imputation, then local CV is upper. 
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

imputation()
