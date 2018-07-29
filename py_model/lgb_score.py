'''
one imputation method, 10 iterations, may take 1 hrs.
'''
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing
import warnings
from utils import init_logging
import logging 

warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(int(time.time()))

CPU_USE_RATE = 0.8
NUM_FOLDS = 5
STRATIFIED = True  
TEST_NULL_HYPO = False
ITERATION = (80 if TEST_NULL_HYPO else 10) # It means how many iterations need to get the final stable AUC score.


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False):
    '''
    num_folds: int, how many foles u'r going to split.

    Maybe we can write a helper function, to find a best parametres each time when u add a new features, to make sure reliability of experiment.
    But, the experiement time will go up more.
    '''
    #---------------------
    # Divide in training/validation and test data
    #---------------------
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    #---------------------
    # core
    #---------------------
    logging.info("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=int(time.time()))
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=int(time.time()))
    # Create arrays and dataframes to store results
    # train
    oof_preds = np.zeros(train_df.shape[0])
    train_preds = np.zeros(train_df.shape[0])
    # test
    sub_preds = np.zeros(test_df.shape[0])
    # feature importance
    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    if TEST_NULL_HYPO:
        train_df['TARGET'] = train_df['TARGET'].copy().sample(frac = 1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        if TEST_NULL_HYPO:
            clf = LGBMClassifier(
                nthread=int(multiprocessing.cpu_count()*CPU_USE_RATE),
                n_estimators=10000,
                learning_rate=0.02,
                num_leaves=127,
                max_depth=8,
                silent=-1,
                verbose=-1,
                random_state=int(time.time()),
                )
        else:
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
                random_state=int(time.time()),
                )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= False, early_stopping_rounds= 100) # early_stopping_rounds= 200
        # training/validating
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        # testing
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        logging.info('Fold %2d val AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    logging.info('Over-folds train AUC score : {}'.format(roc_auc_score(train_df['TARGET'], train_preds)))
    
    over_folds_val_auc = roc_auc_score(train_df['TARGET'], oof_preds)
    logging.info('Over-folds val AUC score : {}'.format(over_folds_val_auc))
    
    # # Write submission file and plot feature importance
    # test_df.loc[:,'TARGET'] = sub_preds
    # test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    return feature_importance_df, over_folds_val_auc

def main():
    #--------------------
    # load features
    #--------------------
    df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
    logging.info('loading features: base_featurs')
    #--------------------
    # out-of-fold validating stratigy + LGB
    #--------------------    
    with timer("Run LightGBM with kfold"):
        feature_importance_df = pd.DataFrame()
        over_folds_val_auc_list = np.zeros(ITERATION)
        for i in range(ITERATION):
            logging.info('Iteration %i' %i)
            iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df, num_folds= NUM_FOLDS, stratified= STRATIFIED)
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_folds_val_auc_list[i] = over_folds_val_auc

        logging.info('Over-iterations val AUC score : {}'.format(over_folds_val_auc_list.mean()))
        logging.info('Standard deviation : {}'.format(over_folds_val_auc_list.std()))
  
        # display_importances(feature_importance_df)
        feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
        useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
        feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

        if TEST_NULL_HYPO:
            feature_importance_df_mean.to_csv("feature_importance-null_hypo.csv", index = True)
        else:
            feature_importance_df_mean.to_csv("feature_importance.csv", index = True)
            useless_features_list = useless_features_df.index.tolist()
            logging.info('useless/overfitting features: \'' + '\', \''.join(useless_features_list) + '\'')


if __name__ == "__main__":
    log_dir = '../log'
    init_logging(log_dir)
    with timer("Lightgbm run a score"):
        main()

