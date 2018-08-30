# source 1: https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# source 2: https://www.kaggle.com/ogrellier/lighgbm-with-selected-features
# source 3: https://github.com/neptune-ml/open-solution-home-credit
# different fold causes at least 0.003: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332

NUM_FOLDS = 5

STRATIFIED = True
TEST_NULL_HYPO = False
ITERATION = (80 if TEST_NULL_HYPO else 1)
USE_SELECTED_VAL = False

SEED = 90210
DEBUG = True
HEAD = 1000

DEVICE='cpu'
BOOSTING_TYPE='gbdt'
OBJECTIVE='binary'
METRIC='auc'
N_ESTIMATORS=5000
LEARNING_RATE=0.02
MAX_BIN=300
MAX_DEPTH=-1
NUM_LEAVES = 30
MIN_CHILD_SAMPLES=70
SUBSAMPLE=1.0
SUBSAMPLE_FREQ=1
COLSAMPLE_BYTREE=0.05
MIN_SPLIT_GAIN=0.5
REG_ALPHA=0
REG_LAMBDA=100
SCALE_POS_WEIGHT=1
VERBOSE=1
SILENT=-1

NUM_WORKERS=29
EARLY_STOPPING_ROUNDS=100

installments__last_k_trend_periods=[10, 50, 100, 500]
installments__last_k_agg_periods=[1, 5, 10, 20, 50, 100]
installments__last_k_agg_period_fractions=[(5,20),(5,50),(10,50),(10,100),(20,100)]
pos_cash__last_k_trend_periods=[6, 12]
pos_cash__last_k_agg_periods=[6, 12, 30]
numbers_of_previous_applications = [1, 2, 3, 4, 5]

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import multiprocessing as mp
from os.path import exists
from functools import partial
from tqdm import tqdm
from scipy.stats import kurtosis, iqr, skew
import category_encoders as ce
warnings.simplefilter(action='ignore', category=FutureWarning)

FEATURE_GRAVEYARD = [
    # useless columns, only "FLAG_DOCUMENT_3, 4, 5, 6, 7, 8, 9, 11, 18" left
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',

    # highly correlated numerical columns, only "FLAG_EMP_PHONE" left
    'AMT_GOODS_PRICE', 'APARTMENTS_MEDI', 'APARTMENTS_MODE', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_MODE', 'COMMONAREA_MEDI',
    'COMMONAREA_MODE', 'ELEVATORS_MEDI', 'ELEVATORS_MODE', 'ENTRANCES_MEDI', 'ENTRANCES_MODE', 'FLOORSMAX_MEDI',
    'FLOORSMAX_MODE', 'FLOORSMIN_MEDI', 'FLOORSMIN_MODE', 'LANDAREA_MEDI', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
    'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MEDI', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE',
    'NONLIVINGAREA_MEDI', 'NONLIVINGAREA_MODE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'REGION_RATING_CLIENT_W_CITY',
    'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MEDI', 'YEARS_BUILD_MODE',
]
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True, encoding = 'ordinal'):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    binary_categorical_columns = [col for col in categorical_columns if len(pd.unique(df[col].dropna())) <= 2]
    for bin in binary_categorical_columns:
        df[bin+'_01encoded'], uniques = pd.factorize(df[bin])
    
    nonbinary_categorical_columns = list(set(categorical_columns) - set(binary_categorical_columns))
    
    new_categorical_columns = []

    if (len(nonbinary_categorical_columns) > 0):
        encoded = pd.get_dummies(df[nonbinary_categorical_columns], dummy_na= nan_as_category)
        df = pd.concat([df, encoded], axis=1)
        new_categorical_columns = [c for c in encoded.columns if c not in original_columns]
    
    return df, new_categorical_columns


def drop_categorical_columns(df):
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = df.drop(categorical_columns, axis= 1)
    return df

def read_df(name):
    if exists('../input/%s.h5' %name):
        df = pd.read_hdf('../input/%s.h5' %name, str(name))
    else:
        df = pd.read_csv('../input/%s.csv' %name)
        df.to_hdf('../input/%s.h5' %name, str(name))
    return df

def fillna_with_gaussian(df):
    a = df.values
    m = np.isnan(a) # mask of NaNs
    a[m] = np.random.normal(df.mean(), df.std(), size=m.sum())
    return df

def group_target_by_cols(df, recipe, base_id='SK_ID_CURR'):
    features = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'].unique()})
    for m in range(len(recipe)):
        group_id_cols = recipe[m][0]
        for n in range(len(recipe[m][1])):
            select = recipe[m][1][n][0]
            method = recipe[m][1][n][1]
            name_grouped_target = '_'.join(group_id_cols)+'_'+method+'_'+select
            tmp = df[group_id_cols + [select]].groupby(group_id_cols).agg(method)
            tmp = tmp.reset_index()
            tmp.columns = pd.Index(group_id_cols+[name_grouped_target])

            # df = df.merge(tmp, how='left', on=group_id_cols)
            features = features.merge(tmp, how='left', on=group_id_cols)

    return features

def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0

def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_

def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features

cols_to_agg = ['AMT_CREDIT', 
               'AMT_ANNUITY',
               'AMT_INCOME_TOTAL',
               'AMT_GOODS_PRICE', 
               'EXT_SOURCE_1',
               'EXT_SOURCE_2',
               'EXT_SOURCE_3',
               'OWN_CAR_AGE',
               'REGION_POPULATION_RELATIVE',
               'DAYS_REGISTRATION',
               'CNT_CHILDREN',
               'CNT_FAM_MEMBERS',
               'DAYS_ID_PUBLISH',
               'DAYS_BIRTH',
               'DAYS_EMPLOYED'
]
aggs = ['min', 'mean', 'max', 'sum', 'var']
aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

APPLICATION_AGGREGATION_RECIPIES = [
    (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
    (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
    (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),
    # to do
    # (['CODE_GENDER', 'OCCUPATION_TYPE'], [
    #     ('AMT_ANNUITY', 'mean'),
    #     ('CNT_CHILDREN', 'mean'),
    #     ('CNT_FAM_MEMBERS', 'mean'),
    #     ('DAYS_BIRTH', 'mean'),
    #     ('DAYS_EMPLOYED', 'mean'),
    #     ('DAYS_ID_PUBLISH', 'mean'),
    #     ('DAYS_REGISTRATION', 'mean'),
    #     ('EXT_SOURCE_1', 'mean'),
    #     ('EXT_SOURCE_2', 'mean'),
    #     ('EXT_SOURCE_3', 'mean'),
    # ]),
    # (['REG_CITY_NOT_WORK_CITY', 'OCCUPATION_TYPE'], [
    #     ('AMT_ANNUITY', 'mean'),
    #     ('CNT_CHILDREN', 'mean'),
    #     ('CNT_FAM_MEMBERS', 'mean'),
    #     ('DAYS_BIRTH', 'mean'),
    #     ('DAYS_EMPLOYED', 'mean'),
    #     ('DAYS_ID_PUBLISH', 'mean'),
    #     ('DAYS_REGISTRATION', 'mean'),
    #     ('EXT_SOURCE_1', 'mean'),
    #     ('EXT_SOURCE_2', 'mean'),
    #     ('EXT_SOURCE_3', 'mean'),
    # ]),
    # (['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], [
    #     ('AMT_ANNUITY', 'mean'),
    #     ('DAYS_BIRTH', 'mean'),
    #     ('DAYS_EMPLOYED', 'mean'),
    #     ('EXT_SOURCE_1', 'mean'),
    #     ('AMT_CREDIT', 'mean'),
    #     ('AMT_GOODS_PRICE', 'mean'),
    #     ('AMT_INCOME_TOTAL', 'mean'),
    # ]),
    # (['ORGANIZATION_TYPE', 'OCCUPATION_TYPE'], [
    #     ('AMT_ANNUITY', 'mean'),
    #     ('CNT_CHILDREN', 'mean'),
    #     ('CNT_FAM_MEMBERS', 'mean'),
    #     ('DAYS_BIRTH', 'mean'),
    #     ('DAYS_EMPLOYED', 'mean'),
    #     ('DAYS_ID_PUBLISH', 'mean'),
    #     ('DAYS_REGISTRATION', 'mean'),
    #     ('EXT_SOURCE_1', 'mean'),
    #     ('EXT_SOURCE_2', 'mean'),
    #     ('EXT_SOURCE_3', 'mean'),
    # ]),
]

# Preprocess application_train.csv and application_test.csv
def application_train_test(nan_as_category = False):
    # Read data and merge
    df = read_df('application_train')
    test_df = read_df('application_test')
    if DEBUG:
        df = df[:HEAD]
        test_df = test_df[:HEAD]
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    
    df = featrue_extraction_application(df)
    test_df = featrue_extraction_application(test_df)

    df = df.append(test_df).reset_index()

    # to do: Categorical features with One-Hot encode
    # df, new_application_categorical_cols = one_hot_encoder(df, nan_as_category)

    del test_df
    gc.collect()
    return df

def featrue_extraction_application(df):

    # cleaning
    df['CODE_GENDER'].replace('XNA',np.nan, inplace=True) # df = df[df['CODE_GENDER'] != 'XNA']
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

    # docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    # # df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    # df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    # live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f)]
    # df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    # df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    # inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    # df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['annuity_income_percentage'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    # df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # df['INCOME_TO_GOODS_PRICE_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_GOODS_PRICE']
    # df['DIFF_INCOME_AND_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']
    df['car_to_birth_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['car_to_employ_ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['children_ratio'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['credit_to_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['credit_to_goods_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE'] # df['DIFF_CREDIT_AND_GOODS'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['days_employed_percentage'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['income_credit_percentage'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['income_per_child'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['payment_rate'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['phone_to_birth_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['phone_to_employ_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['external_sources_weighted'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4
    df['cnt_non_child'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['child_to_non_child_ratio'] = df['CNT_CHILDREN'] / df['cnt_non_child']
    df['income_per_non_child'] = df['AMT_INCOME_TOTAL'] / df['cnt_non_child']
    df['credit_per_person'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['credit_per_child'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    df['credit_per_non_child'] = df['AMT_CREDIT'] / df['cnt_non_child']
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']: # replace NEW_SCORES_STD
        df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    df['short_employment'] = (df['DAYS_EMPLOYED'] < -2000).astype(int) # should be long_employment
    df['young_age'] = (df['DAYS_BIRTH'] < -14000).astype(int) # should be retirement_age


    # todo: use "istest" is worse?
    # df = df.merge(pd.read_csv('../util/istest_pred/istest_pred-linux2.csv'), how='left', on=['SK_ID_CURR'])  

    # df['EXT_SOURCE_1_VAR'] = (df['EXT_SOURCE_1'] - df['NEW_EXT_SOURCES_MEAN'])**2
    # df['EXT_SOURCE_2_VAR'] = (df['EXT_SOURCE_2'] - df['NEW_EXT_SOURCES_MEAN'])**2
    # df['EXT_SOURCE_3_VAR'] = (df['EXT_SOURCE_3'] - df['NEW_EXT_SOURCES_MEAN'])**2
    # df['EXT_SOURCE_1_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_1_VAR'].median())
    # df['EXT_SOURCE_2_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_2_VAR'].median())
    # df['EXT_SOURCE_3_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_3_VAR'].median())

    # df['SOCIAL_PRED'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['DEF_30_CNT_SOCIAL_CIRCLE'] + 2*df['OBS_60_CNT_SOCIAL_CIRCLE'] + 2*df['DEF_60_CNT_SOCIAL_CIRCLE']


    # grouping categorical features to generate new features
    cols_to_remove = []
    for group_id_cols, select_and_method in APPLICATION_AGGREGATION_RECIPIES:
        for select, method in select_and_method:
            name_grouped_target = '_'.join(group_id_cols)+'_'+method+'_'+select
            tmp = df[group_id_cols + [select]].groupby(group_id_cols).agg(method)
            tmp = tmp.reset_index()
            tmp.columns = pd.Index(group_id_cols+[name_grouped_target])
            df = df.merge(tmp, how='left', on=group_id_cols)
            
            ## to do: redundant processings
            cols_to_remove.append(name_grouped_target)
            if method in ['mean', 'median', 'max', 'min']:
                diff_feature_name = name_grouped_target+'_diff'
                abs_diff_feature_name = name_grouped_target+'_abs_diff'

                df[diff_feature_name] = df[select] - df[name_grouped_target]
                df[abs_diff_feature_name] = np.abs(df[select] - df[name_grouped_target])

    df.drop(cols_to_remove, axis=1, inplace= True)
    return df

BUREAU_NUM_AGGREGATION_RECIPIES = [
    ('CREDIT_TYPE', 'count'),
    ('CREDIT_ACTIVE', 'size')
]
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in [
        'DAYS_CREDIT',
        'AMT_ANNUITY',
        'AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT',
        'AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM_OVERDUE',
        'AMT_CREDIT_MAX_OVERDUE',
        'CNT_CREDIT_PROLONG',
        'CREDIT_DAY_OVERDUE',
        'DAYS_CREDIT_ENDDATE',
        'DAYS_CREDIT_UPDATE'
    ]:
        BUREAU_NUM_AGGREGATION_RECIPIES.append((select, agg))
BUREAU_NUM_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_NUM_AGGREGATION_RECIPIES)]

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(nan_as_category = True):
    bureau = read_df('bureau')

    bb = read_df('bureau_balance')
    if DEBUG:
        bureau = bureau[:HEAD]
        bb = bb[:HEAD]

    bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    
    # to do
    # bureau_balance, bureau categorical features
    # bb, new_bb_categorical_cols = one_hot_encoder(bb, nan_as_category)
    # bureau, new_bureau_categorical_cols = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    
    # to do
    # for col in new_bb_categorical_cols:
    #     bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    # merge bureau_balance.csv with bureau.csv
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    del bb, bb_agg
    gc.collect()

    # to do: fill with median?
    # if FILL_MISSING:
    #     bureau['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
    #     bureau['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
    #     bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
    #     bureau['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)
    

    # to do
    # for cat in new_bureau_categorical_cols: BUREAU_NUM_AGGREGATION_RECIPIES[0][1].append((cat, 'mean')) # cat_aggregations[cat + "_MEAN"]
    # for cat in new_bb_categorical_cols: BUREAU_NUM_AGGREGATION_RECIPIES[0][1].append((cat+'_MEAN', 'mean'))

    bureau_agg = group_target_by_cols(bureau, BUREAU_NUM_AGGREGATION_RECIPIES)
    bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
    bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)

    groupby = bureau.groupby(by=['SK_ID_CURR'])

    g = groupby['DAYS_CREDIT'].agg('count').reset_index()
    g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
    g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_active_binary': 'bureau_credit_active_binary'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
    g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
    bureau_agg = bureau_agg.merge(g, on=['SK_ID_CURR'], how='left')

    bureau_agg['bureau_average_of_past_loans_per_type'] = bureau_agg['bureau_number_of_past_loans'] / bureau_agg['bureau_number_of_loan_types']

    bureau_agg['bureau_debt_credit_ratio'] = bureau_agg['bureau_total_customer_debt'] / bureau_agg['bureau_total_customer_credit']

    bureau_agg['bureau_overdue_debt_ratio'] = bureau_agg['bureau_total_customer_overdue'] / bureau_agg['bureau_total_customer_debt']

    del bureau
    gc.collect()
    return bureau_agg


CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_BALANCE',
                   'AMT_CREDIT_LIMIT_ACTUAL',
                   'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT',
                   'AMT_DRAWINGS_OTHER_CURRENT',
                   'AMT_DRAWINGS_POS_CURRENT',
                   'AMT_PAYMENT_CURRENT',
                   'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT',
                   'CNT_DRAWINGS_OTHER_CURRENT',
                   'CNT_INSTALMENT_MATURE_CUM',
                   'MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]

# Preprocess credit_card_balance.csv
def credit_card_balance(nan_as_category = True):
    credit_card = read_df('credit_card_balance')
    if DEBUG:
        credit_card = credit_card[:HEAD]

    # to do
    # credit_card, new_credit_card_categorical_cols = one_hot_encoder(credit_card, nan_as_category= True)

    credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

    # to do
    # for cat in new_credit_card_categorical_cols: CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES[0][1].append((cat, 'mean')) 

    credit_card_agg = group_target_by_cols(credit_card, CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)

    # static features
    credit_card['number_of_installments'] = credit_card.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()['CNT_INSTALMENT_MATURE_CUM']
    credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

    groupby = credit_card.groupby(by=['SK_ID_CURR'])

    g = groupby['SK_ID_PREV'].agg('nunique').reset_index() # credit_card_agg['CC_COUNT'] = credit_card.groupby('SK_ID_CURR').size()
    g.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['SK_DPD'].agg('mean').reset_index()
    g.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['number_of_installments'].agg('sum').reset_index()
    g.rename(index=str, columns={'number_of_installments': 'credit_card_total_installments'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
    g.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'}, inplace=True)
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')

    credit_card_agg['credit_card_cash_card_ratio'] = credit_card_agg['credit_card_drawings_atm'] / credit_card_agg['credit_card_drawings_total']

    credit_card_agg['credit_card_installments_per_loan'] = credit_card_agg['credit_card_total_installments'] / credit_card_agg['credit_card_number_of_loans']


    # dynamic features
    credit_card_sorted = credit_card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    credit_card_sorted['credit_card_monthly_diff'] = credit_card_sorted.groupby(by=['SK_ID_CURR'])['AMT_BALANCE'].diff()
    g = credit_card_sorted.groupby(by=['SK_ID_CURR'])['credit_card_monthly_diff'].agg('mean').reset_index()
    credit_card_agg = credit_card_agg.merge(g, on=['SK_ID_CURR'], how='left')


    credit_card_agg.drop(['SK_ID_PREV'], axis= 1, inplace = True, errors='ignore')

    del credit_card
    gc.collect()
    return credit_card_agg


POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

# Preprocess POS_CASH_balance.csv
def pos_cash(nan_as_category = True):
    pos_cash = read_df('POS_CASH_balance')
    if DEBUG:
        pos_cash = pos_cash[:HEAD]

    # to do
    # pos_cash, new_pos_cash_categorical_cols = one_hot_encoder(pos_cash, nan_as_category= True)
    # for cat in new_pos_cash_categorical_cols:
        # POS_CASH_BALANCE_AGGREGATION_RECIPIES[0][1].append((cat, 'mean'))

    pos_cash_agg = group_target_by_cols(pos_cash, POS_CASH_BALANCE_AGGREGATION_RECIPIES)

    pos_cash['is_contract_status_completed'] = pos_cash['NAME_CONTRACT_STATUS'] == 'Completed'
    pos_cash['pos_cash_paid_late'] = (pos_cash['SK_DPD'] > 0).astype(int)
    pos_cash['pos_cash_paid_late_with_tolerance'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)

    func = partial(
        pos_cash__generate_features,
        agg_periods=pos_cash__last_k_agg_periods,
        trend_periods=pos_cash__last_k_trend_periods
    )
    g = parallel_apply(pos_cash.groupby(['SK_ID_CURR']), func, index_name='SK_ID_CURR', num_workers=NUM_WORKERS).reset_index()
    g['pos_cash_remaining_installments'] = g['pos_cash_remaining_installments'].astype('float64')
    pos_cash_agg = pos_cash_agg.merge(g, on='SK_ID_CURR', how='left')

    # to do
    # pos['TIME_DECAYED_UNPAYED_RATIO'] = pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT'] / pos['MONTHS_BALANCE']**2

    # Count pos cash accounts # to do
    # pos_cash_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()

    del pos_cash
    gc.collect()
    return pos_cash_agg

def pos_cash__generate_features(gr, agg_periods, trend_periods):
    one_time = pos_cash__one_time_features(gr)
    all = pos_cash__all_installment_features(gr)
    agg = pos_cash__last_k_installment_features(gr, agg_periods)
    trend = pos_cash__trend_in_last_k_installment_features(gr, trend_periods)
    last = pos_cash__last_loan_features(gr)
    features = {**one_time, **all, **agg, **trend, **last}

    return pd.Series(features)

def pos_cash__one_time_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
    features = {}

    features['pos_cash_remaining_installments'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1)
    features['pos_cash_completed_contracts'] = gr_['is_contract_status_completed'].agg('sum')

    return features

def pos_cash__all_installment_features(gr):
    return pos_cash__last_k_installment_features(gr, periods=[10e16])

def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features

def pos_cash__last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                         ['count', 'mean'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                         ['count', 'mean'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                         period_name)
    return features

def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features

def pos_cash__trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_trend_feature(features, gr_period,
                                     'SK_DPD', '{}_period_trend_'.format(period)
                                     )
        features = add_trend_feature(features, gr_period,
                                     'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                     )
        features = add_trend_feature(features, gr_period,
                                     'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period)
                                     )
    return features

def pos_cash__last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

    features={}
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                     ['count', 'sum', 'mean'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                     ['mean'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD',
                                     ['sum', 'mean', 'max', 'std'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                     ['sum', 'mean', 'max', 'std'],
                                     'last_loan_')

    return features

PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

# Preprocess previous_applications.csv
def previous_applications(nan_as_category = True):
    prev_app = read_df('previous_application')
    if DEBUG:
        prev_app = prev_app[:HEAD]

    # to do
    # prev_app, new_prev_app_categorical_cols = one_hot_encoder(prev_app, nan_as_category= True)

    prev_app['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev_app['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev_app['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev_app['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev_app['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # to do
    # for cat in new_prev_app_categorical_cols: PREVIOUS_APPLICATION_AGGREGATION_RECIPIES[0][1].append((cat, 'mean'))
    
    prev_app_agg = group_target_by_cols(prev_app, PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)

    prev_app_sorted = prev_app.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
    prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

    prev_app_sorted['previous_application_prev_was_approved'] = (prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    g = prev_app_sorted_groupby['previous_application_prev_was_approved'].last().reset_index()
    prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    prev_app_sorted['previous_application_prev_was_refused'] = (prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    g = prev_app_sorted_groupby['previous_application_prev_was_refused'].last().reset_index()
    prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
    g.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
    prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    g = prev_app_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_refused'].mean().reset_index()
    g.rename(index=str, columns={'previous_application_prev_was_refused': 'previous_application_fraction_of_refused_applications'},inplace=True)
    prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    prev_app_sorted['prev_application_prev_was_revolving_loan'] = (prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
    g = prev_app_sorted.groupby(by=['SK_ID_CURR'])['prev_application_prev_was_revolving_loan'].last().reset_index()
    prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    for number in numbers_of_previous_applications:
        prev_applications_tail = prev_app_sorted_groupby.tail(number)

        tail_groupby = prev_applications_tail.groupby(by=['SK_ID_CURR'])

        g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
        g.rename(index=str, columns={'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)}, inplace=True)
        prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

        g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
        g.rename(index=str, columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)}, inplace=True)
        prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

        g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
        g.rename(index=str, columns={'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)}, inplace=True)
        prev_app_agg = prev_app_agg.merge(g, on=['SK_ID_CURR'], how='left')

    # # Add feature: value ask / value received percentage
    # prev_app['APP_CREDIT_PERC'] = prev_app['AMT_APPLICATION'] / prev_app['AMT_CREDIT']
    # prev_app['HUMAN_EVAL'] = prev_app['NAME_CONTRACT_STATUS_Approved'] - prev_app['NAME_CONTRACT_STATUS_Refused']
    # prev_app['TIME_DECAYED_HUMAN_EVAL'] = prev_app['HUMAN_EVAL'] / prev_app['DAYS_DECISION']**2

    gc.collect()
    return prev_app_agg

INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_INSTALMENT',
                   'AMT_PAYMENT',
                   'DAYS_ENTRY_PAYMENT',
                   'DAYS_INSTALMENT',
                   'NUM_INSTALMENT_NUMBER',
                   'NUM_INSTALMENT_VERSION'
                   ]:
        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

# Preprocess installments_payments.csv
def installments_payments(df, nan_as_category = True):
    installments = read_df('installments_payments')
    installments = installments[:HEAD]

    # to do
    # installments, new_installments_categorical_cols = one_hot_encoder(installments, nan_as_category= True)
    # for cat in new_installments_categorical_cols:
    #     INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES[0][1].append((cat, 'mean'))

    installments_agg = group_target_by_cols(installments, INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)
    installments['installment_paid_late_in_days'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['installment_paid_late'] = (installments['installment_paid_late_in_days'] > 0).astype(int)
    installments['installment_paid_over_amount'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
    installments['installment_paid_over'] = (installments['installment_paid_over_amount'] > 0).astype(int)

    groupby = installments.groupby(['SK_ID_CURR'])
    func = partial(installments_payments__generate_features,
                   agg_periods=installments__last_k_agg_periods,
                   period_fractions=installments__last_k_agg_period_fractions,
                   trend_periods=installments__last_k_trend_periods)
    g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=NUM_WORKERS).reset_index()

    installments_agg = installments_agg.merge(g, on='SK_ID_CURR', how='left')

    # # Percentage and difference paid in each installment (amount paid and installment value)
    # installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    # installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    # # Days past due and days before due (no negative values)
    # installments['DPD'] = (installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']) / installments['DAYS_INSTALMENT']**2
    # installments['DBD'] = (installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']) / installments['DAYS_INSTALMENT']**2
    # installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    # installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)

    # # Count installments accounts
    # installments_agg['INSTAL_COUNT'] = installments.groupby('SK_ID_CURR').size()


    del installments
    gc.collect()
    return installments_agg

def installments_payments__generate_features(gr, agg_periods, trend_periods, period_fractions):
    all = installments_payments__all_installment_features(gr)
    agg = installments_payments__last_k_installment_features_with_fractions(gr, agg_periods, period_fractions)
    trend = installments_payments__trend_in_last_k_installment_features(gr, trend_periods)
    last = installments_payments__last_loan_features(gr)
    features = {**all, **agg, **trend, **last}
    return pd.Series(features)

def installments_payments__all_installment_features(gr):
    return installments_payments__last_k_installment_features(gr, periods=[10e16])

def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])

def installments_payments__last_k_installment_features_with_fractions(gr, periods, period_fractions):
    features = installments_payments__last_k_installment_features(gr, periods)

    for short_period, long_period in period_fractions:
        short_feature_names = get_feature_names_by_period(features, short_period)
        long_feature_names = get_feature_names_by_period(features, long_period)

        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
    return features

def installments_payments__last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         period_name)

        features = add_features_in_group(features, gr_period, 'installment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'installment_paid_late',
                                         ['count', 'mean'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'installment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'installment_paid_over',
                                         ['count', 'mean'],
                                         period_name)
    return features

def installments_payments__trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_trend_feature(features, gr_period,
                                     'installment_paid_late_in_days', '{}_period_trend_'.format(period)
                                     )
        features = add_trend_feature(features, gr_period,
                                     'installment_paid_over_amount', '{}_period_trend_'.format(period)
                                     )
    return features

def installments_payments__last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

    features = {}
    features = add_features_in_group(features, gr_,
                                     'installment_paid_late_in_days',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_,
                                     'installment_paid_late',
                                     ['count', 'mean'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_,
                                     'installment_paid_over_amount',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_,
                                     'installment_paid_over',
                                     ['count', 'mean'],
                                     'last_loan_')
    return features

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, seed = int(time.time())):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    
    # print all features
    print('All features:', train_df.columns.tolist())
    print('e.g., for some examplary guys 100193:')
    print(train_df.loc[train_df['SK_ID_CURR']==100193].values.tolist())
    # exit()

    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    train_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    if TEST_NULL_HYPO:
        train_df['TARGET'] = train_df['TARGET'].copy().sample(frac=1.0).values
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        if TEST_NULL_HYPO:
            clf = LGBMClassifier(
                device=DEVICE,
                boosting_type=BOOSTING_TYPE,
                objective=OBJECTIVE,
                metric=METRIC,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                max_bin=MAX_BIN,
                max_depth=MAX_DEPTH,
                num_leaves=NUM_LEAVES,
                min_child_samples=MIN_CHILD_SAMPLES,
                subsample=SUBSAMPLE,
                subsample_freq=SUBSAMPLE_FREQ,
                colsample_bytree=COLSAMPLE_BYTREE,
                min_split_gain=MIN_SPLIT_GAIN,
                reg_alpha=REG_ALPHA,
                reg_lambda=REG_LAMBDA,
                scale_pos_weight = SCALE_POS_WEIGHT,

                nthread=NUM_WORKERS,
                silent=SILENT,
                verbose=VERBOSE,
                random_state=seed,
                )
        else:
            clf = LGBMClassifier(
                device=DEVICE,
                boosting_type=BOOSTING_TYPE,
                objective=OBJECTIVE,
                metric=METRIC,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                max_bin=MAX_BIN,
                max_depth=MAX_DEPTH,
                num_leaves=NUM_LEAVES,
                min_child_samples=MIN_CHILD_SAMPLES,
                subsample=SUBSAMPLE,
                subsample_freq=SUBSAMPLE_FREQ,
                colsample_bytree=COLSAMPLE_BYTREE,
                min_split_gain=MIN_SPLIT_GAIN,
                reg_alpha=REG_ALPHA,
                reg_lambda=REG_LAMBDA,
                scale_pos_weight = SCALE_POS_WEIGHT,

                nthread=NUM_WORKERS,
                silent=SILENT,
                verbose=VERBOSE,
                random_state=seed,
                )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= False, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        train_preds[train_idx] += clf.predict_proba(train_x, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d val AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('---------------------------------------\nOver-folds train AUC score %.6f' % roc_auc_score(train_df['TARGET'], train_preds))
    over_folds_val_auc = roc_auc_score(train_df['TARGET'], oof_preds)
    print('Over-folds val AUC score %.6f\n---------------------------------------' % over_folds_val_auc)
    # Write submission file and plot feature importance
    test_df.loc[:,'TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    return feature_importance_df, over_folds_val_auc

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('lgbm_importances.png')

def rename_columns(df, name):
    for col in df.columns:
        if col != 'SK_ID_CURR':
            df.rename(index=str, columns={col: name+'_'+col}, inplace=True)
    return df

def main():
    df = pd.DataFrame()
    with timer("Process train/test application"):
        df = application_train_test()

        print("Train/Test application df shape:", df.shape)
        # print("with features:", df.columns.tolist())        
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance()
        bureau = rename_columns(bureau, 'bureau')

        print("Bureau df shape:", bureau.shape)
        # print("with features:", bureau.columns.tolist())
        df = df.merge(bureau, how='left', on=['SK_ID_CURR'])
        del bureau
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance()
        cc = rename_columns(cc, 'credit_card_balance')

        print("Credit card balance df shape:", cc.shape)
        # print("with features:", cc.columns.tolist())
        df = df.merge(cc, how='left', on=['SK_ID_CURR'])
        del cc
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(df)
        ins = rename_columns(ins, 'installments_payments')

        print("Installments payments df shape:", ins.shape)
        # print("with features:", ins.columns.tolist())
        df = df.merge(ins, how='left', on=['SK_ID_CURR'])
        del ins
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash()
        pos = rename_columns(pos, 'POS_CASH_balance')

        print("Pos-cash balance df shape:", pos.shape)
        # print("with features:", pos.columns.tolist())
        df = df.merge(pos, how='left', on=['SK_ID_CURR'])

        del pos
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications()
        prev = rename_columns(prev, 'previous_application')
    
        print("Previous applications df shape:", prev.shape)
        # print("with features:", prev.columns.tolist())
        df = df.merge(prev, how='left', on=['SK_ID_CURR'])
        del prev
        gc.collect()
    with timer("Run LightGBM with kfold"):

        df['nan_count'] = df.isnull().sum(axis=1)

        print(df.shape)
        df.drop(FEATURE_GRAVEYARD, axis=1, inplace=True, errors='ignore')
        gc.collect()
        print(df.shape)
      
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        encoder = ce.OrdinalEncoder(cols=categorical_columns,drop_invariant=False) # to do: True, then 1218 featuers -> 998 features??
        # print('categorical columns:', categorical_columns)

        encoder.fit(df)
        df = encoder.transform(df)

        feature_importance_df = pd.DataFrame()
        over_iterations_val_auc = np.zeros(ITERATION)
        for i in range(ITERATION):
            print('Iteration %i' %i)
            iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df, num_folds = NUM_FOLDS, stratified = STRATIFIED, seed = SEED)
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_iterations_val_auc[i] = over_folds_val_auc

        print('============================================\nOver-iterations val AUC score %.6f' %over_iterations_val_auc.mean())
        print('Standard deviation %.6f\n============================================' %over_iterations_val_auc.std())
        
        # display_importances(feature_importance_df)
        feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
        useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
        feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

        if TEST_NULL_HYPO:
            feature_importance_df_mean.to_csv("feature_importance-null_hypo.csv", index = True)
        elif not DEBUG:
            feature_importance_df_mean.to_csv("feature_importance.csv", index = True)
            useless_features_list = useless_features_df.index.tolist()
            print('Useless features: \'' + '\', \''.join(useless_features_list) + '\'')

if __name__ == "__main__":
    submission_file_name = "sub.csv"
    with timer("Full model run"):
        main()