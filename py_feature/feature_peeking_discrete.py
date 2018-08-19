import pandas as pd
import numpy as np
import pandas as pd
import time
import os
import multiprocessing as mp # for speeding up some process
import sys
sys.path.append('../py_model')
from utils import init_logging
import logging 
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import multiprocessing
from sklearn.metrics import roc_auc_score
from operator import itemgetter


# base features
df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()
no_need_to_inpute = ['index']
df.drop(no_need_to_inpute, axis = 1, inplace = True)
print ('df', df.shape)
# find discrete features
discrete_features = []
continuous_features = []
for f in df.columns.tolist()[:]:
    if df[f].value_counts().size < 20:
        discrete_features.append(f)
    else:
        continuous_features.append(f)
print ('discrete_features : {}'.format(len(discrete_features)))
print ('continuous_features : {}'.format(len(continuous_features)))
# check 
print ('SK_ID_CURR' in discrete_features, 'TARGET' in discrete_features, 'index' in discrete_features)


# setting
num_folds = 5
CPU_USE_RATE = 0.4
df = df[discrete_features]
print ('df',df.shape)
train_df = df[df['TARGET'].notnull()]
print ('train', train_df.shape)
test_df = df[df['TARGET'].isnull()]
print ('test', test_df.shape)
# output
oof_preds = np.zeros(train_df.shape[0]) # substitue the target of training part in df
print ('oof_preds', oof_preds.shape)
train_preds = np.zeros(train_df.shape[0])
print ('train_preds', train_preds.shape)
# test
sub_preds = np.zeros(test_df.shape[0]) # substitue the target of testing part in df
print ('sub_preds', sub_preds.shape)
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
# data peeking
folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=int(time.time()))
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
    s = time.time()
    print ('n_fold', n_fold)
    train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
    print ('train_x',train_x.shape)
    #print ('train_idx',train_idx)
    valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    print ('valid_x',valid_x.shape)
    #print ('valid_idx',valid_idx)
    # ratio = pd.Series(train_y).value_counts().iloc[0] / pd.Series(train_y).value_counts().iloc[1] # > 1
    # print ('ratio', ratio)
    #----------------
    # model 
    #----------------
    clf = LGBMClassifier(
        nthread=int(multiprocessing.cpu_count()*CPU_USE_RATE),
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34, # 20
        colsample_bytree=0.2, #0.9497036 < 0.2
        subsample=0.8715623,
        max_depth=8, # 7
        reg_alpha=0.041545473, # 0.3
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775, # 60
        silent=-1,
        verbose=-1,
        random_state = int(time.time()),
        )
    #----------------
    # training 
    #----------------
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric= 'auc', verbose= False, early_stopping_rounds= 100) # early_stopping_rounds= 100
    # training/validating
    probas_ = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    exp = []
    for th in np.arange(0,1, 0.001):
        y_scores = [1 if p_ >= th else 0 for p_ in probas_]
        exp.append((th, roc_auc_score(valid_y, y_scores)))
    best_th = max(exp, key = itemgetter(1))[0]
    best_performance = max(exp, key = itemgetter(1))[1]
    print ('best_th', best_th)
    print ('best_auc', best_performance) # 0.613309518754671
    oof_preds[valid_idx] = [1 if p_ >= best_th else 0 for p_ in probas_]
    print ('oof_preds', pd.Series(oof_preds[valid_idx]).value_counts()) 
    # testing
    test_prob_ = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
    sub_preds +=  np.array([1 if p_ >= best_th else 0 for p_ in test_prob_])/ folds.n_splits
    e = time.time()
    print (e - s)
#-------
# observe validating auc
#-------
over_folds_val_auc = roc_auc_score(train_df['TARGET'], oof_preds)
print ('Over-folds val AUC score : {}'.format(over_folds_val_auc))

#-------------
# output
#-------------
sub_preds = np.array([1 if i >= 0.5 else 0 for i in sub_preds])
target = np.concatenate((oof_preds, sub_preds), axis = 0)
df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
df = df[discrete_features]
df['TARGET'] = target
df.to_csv('../features/peeking_target_for_imputing_w_validating_{}_auc_v2.csv'.format(over_folds_val_auc), index = False)


