import pandas as pd
import numpy as np
import sys
sys.path.append('../py_model')
from utils import init_logging
import logging 
import gc
from fancyimpute import MICE # for imputing
import os



#---------------------------
# loading feature correlations matrix
#---------------------------
feature_correlation = pd.read_excel('../input/feature_correlation_matrix.xlsx')
feature_correlation.drop(['index','SK_ID_CURR'], axis = 1, inplace = True)
print (feature_correlation.shape)

#----------------------------
# step1: find the pair of features whose has passed our threshold(加速後面挑features的速度)
#----------------------------

# setting
coeifficient_th = 0.9 # the less feature we impute is better
highly_similar_feature_ls = []
pair_of_features = []
for ix, row in feature_correlation.head(n = 500).iterrows():
    hign_similarity = [i for i in row.tolist()  if i > coeifficient_th or i < -coeifficient_th ]
    hign_similarity = [i for i in hign_similarity if i != 1]
    hign_similarity = [i for i in hign_similarity if i != -1]
    if len(hign_similarity)!= 0:
        # find maximum absolute value of list
        max_similarity = max(hign_similarity, key=abs)
        highly_similar_feature = row[row == max_similarity].index[0]
        pair_of_features.append((ix,highly_similar_feature,))
        #print ('max_similarity',max_similarity)
        highly_similar_feature_ls.append(ix)
        highly_similar_feature_ls.append(highly_similar_feature)

highly_similar_feature_ls = list(set(highly_similar_feature_ls))
print ('num pair of highly similar feautures', len(highly_similar_feature_ls))
def pair_similarity(f1, f2):
    '''
    paras:
    f1:str
    f2:str
    '''
    return feature_correlation.iloc[(feature_correlation.index == f1)][[f2]].iloc[0][0]

#----------------------------
# step3: how many base features we can impute
#----------------------------
df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
print (df.shape)
copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()
no_need_to_inpute = ['index','SK_ID_CURR']
df.drop(no_need_to_inpute, axis = 1, inplace = True)
# handling with infinity
df.replace([np.inf, -np.inf], np.nan, inplace = True)
print (df.shape)

# setting
log_dir = '../log_feature_selection'
init_logging(log_dir)

feature_scaling = True
k = 65 # by cv
num_split = 10
logging.info('similarity threshold : {}'.format(coeifficient_th))

for i in np.arange(0.04, 0.001, step = - 0.01):
    ratio_that_similar_to_all_feautres = i
    logging.info('ratio_that_similar_to_all_feautres : {}'.format(ratio_that_similar_to_all_feautres))
    #----------------------------
    # step2: find the subset of features overpass the theeshold with all features from pair of features set
    #----------------------------
    output_features = []
    for e_f in highly_similar_feature_ls:
        check_if_all_features_are_similar = [1.0 if abs(pair_similarity(e_f, e_f1)) > coeifficient_th else 0 for e_f1 in highly_similar_feature_ls ]
        num_features_the_feautre_correlate_w_others = int(len(check_if_all_features_are_similar) * ratio_that_similar_to_all_feautres)
        if check_if_all_features_are_similar.count(1.0) > num_features_the_feautre_correlate_w_others:
            print (e_f)
            output_features.append(e_f)
    num_features_in_raw_features_will_be_impute = len(output_features)
    logging.info('num_features_in_raw_features_will_be_impute: {}'.format(num_features_in_raw_features_will_be_impute))
    base_feature_worth_imputing_set = set(df.columns.tolist()) & set(output_features)
    logging.info('num_base_feature_worth_imputing_set : {}'.format(len(base_feature_worth_imputing_set)))
    #----------------------------
    # loading original features
    #----------------------------
    original_raw_featurs = pd.read_hdf('../features/original_raw_featurs.h5','original_raw_featurs')
    logging.info('original_raw_featurs : {}'.format(original_raw_featurs.shape))

    # feature scaling before KNN
    X_missing_df = original_raw_featurs[output_features]
    logging.info('X_missing_df before feature_scaling : {}'.format(X_missing_df.shape))
    logging.info('feature_scaling: {}'.format(feature_scaling))
    # feature scaling with ignoring np.nan
    if feature_scaling == True:
        for f in X_missing_df.columns.tolist():
            mean = X_missing_df[f].mean()
            std = X_missing_df[f].std()
            X_missing_df[f] = (X_missing_df[f] - mean) / std
    logging.info('X_missing_df after feature_scaling : {}'.format(X_missing_df.shape))
    #----------------------------
    # MICE imputing
    #----------------------------
    if num_features_in_raw_features_will_be_impute != 0:
        # avoid ValueError: Input matrix is not missing any values
        #----------------------------
        # imputing
        #----------------------------
        logging.info('visit_sequence: {}'.format('monotone')) 
        logging.info('impute_type: {}'.format('col')) 
        logging.info('init_fill_method: {}'.format('mean')) 
        
        X_filled = MICE(visit_sequence = 'monotone', 
                        impute_type = 'col',
                        init_fill_method = 'mean').complete(X_missing_df.values)
        X_filled = pd.DataFrame(X_filled, columns = X_missing_df.columns)

        logging.info('X_filled : {}'.format(X_filled.shape))
        #----------------------------
        # reload base_features for filling
        #----------------------------
        df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
        logging.info('base_featurs : {}'.format(df.shape))
        copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()
        no_need_to_inpute = ['index']
        df.drop(no_need_to_inpute, axis = 1, inplace = True)
        for f_in_same_cluster in X_filled.columns.tolist():
            if f_in_same_cluster in set(df.columns.tolist()):
                logging.info('featurs : {}'.format(f_in_same_cluster))
                df[f_in_same_cluster] = X_filled[f_in_same_cluster].tolist()
        #-------------------
        # save
        #-------------------
        output_path = '../features/filled_by_mice'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        df.to_hdf(
            os.path.join(output_path, 'normalized_mice_similar_features_th_09_{}.h5'.format(len(base_feature_worth_imputing_set))), 
            'normalized_mice_similar_features_th_09_{}'.format(len(base_feature_worth_imputing_set)))

        logging.info('done - {}'.format(ratio_that_similar_to_all_feautres))
