# source 1: https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features
# source 2: https://www.kaggle.com/ogrellier/lighgbm-with-selected-features
# different fold causes at least 0.003: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332

NUM_FOLDS = 5
CPU_USE_RATE = 0.8
STRATIFIED = True
TEST_NULL_HYPO = True
ITERATION = (80 if TEST_NULL_HYPO else 10)
RESIDUAl_AGG = False

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
import multiprocessing
from os.path import exists
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(int(time.time()))

feature_graveyard = [
    # importance / null_importance <= 1
    'FLAG_DOCUMENT_21', 'ORGANIZATION_TYPE_Realtor', 'NAME_INCOME_TYPE_Pensioner', 'OCCUPATION_TYPE_IT staff',
    'BURO_CREDIT_CURRENCY_nan_MEAN', 'BURO_CREDIT_CURRENCY_currency 4_MEAN', 'FLAG_CONT_MOBILE', 'BURO_CREDIT_TYPE_Interbank credit_MEAN',
    'BURO_CREDIT_CURRENCY_currency 2_MEAN', 'BURO_CREDIT_CURRENCY_currency 3_MEAN', 'BURO_CREDIT_ACTIVE_nan_MEAN',
    'PREV_WEEKDAY_APPR_PROCESS_START_nan_MEAN', 'ORGANIZATION_TYPE_Postal', 'CC_NAME_CONTRACT_STATUS_Refused_VAR',
    'NAME_INCOME_TYPE_Unemployed', 'FLAG_DOCUMENT_10', 'ORGANIZATION_TYPE_Religion', 'NAME_INCOME_TYPE_Businessman',
    'ORGANIZATION_TYPE_Mobile', 'PREV_PRODUCT_COMBINATION_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_nan_SUM', 'CC_NAME_CONTRACT_STATUS_nan_MIN',
    'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 12', 'ORGANIZATION_TYPE_Industry: type 13',
    'ORGANIZATION_TYPE_Industry: type 2', 'CC_NAME_CONTRACT_STATUS_nan_MEAN', 'ORGANIZATION_TYPE_Industry: type 4',
    'CC_NAME_CONTRACT_STATUS_nan_MAX', 'CC_NAME_CONTRACT_STATUS_Signed_SUM', 'REFUSED_HUMAN_EVAL_MIN', 'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
    'ORGANIZATION_TYPE_Industry: type 7', 'PREV_CHANNEL_TYPE_nan_MEAN', 'NAME_INCOME_TYPE_Maternity leave',
    'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN', 'CC_NAME_CONTRACT_STATUS_nan_VAR', 'NAME_INCOME_TYPE_Student',
    'BURO_CREDIT_CURRENCY_currency 1_MEAN', 'ORGANIZATION_TYPE_Trade: type 5', 'ORGANIZATION_TYPE_Trade: type 4', 'FLAG_DOCUMENT_12',
    'ORGANIZATION_TYPE_Trade: type 1', 'ORGANIZATION_TYPE_Telecom', 'FLAG_DOCUMENT_15', 'CC_NAME_CONTRACT_STATUS_Refused_MAX',
    'PREV_NAME_PORTFOLIO_Cars_MEAN', 'FLAG_DOCUMENT_17', 'PREV_NAME_GOODS_CATEGORY_Direct Sales_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Education_MEAN', 'PREV_NAME_GOODS_CATEGORY_Fitness_MEAN', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
    'PREV_NAME_GOODS_CATEGORY_House Construction_MEAN', 'PREV_NAME_GOODS_CATEGORY_Insurance_MEAN', 'ORGANIZATION_TYPE_Trade: type 6',
    'BURO_CREDIT_TYPE_Loan for the purchase of equipment_MEAN', 'NAME_TYPE_SUITE_Group of people',
    'POS_NAME_CONTRACT_STATUS_Canceled_MEAN', 'BURO_CREDIT_TYPE_Loan for purchase of shares (margin lending)_MEAN',
    'BURO_CREDIT_TYPE_Mobile operator loan_MEAN', 'PREV_NAME_SELLER_INDUSTRY_MLM partners_MEAN', 'PREV_NAME_PORTFOLIO_nan_MEAN',
    'CC_NAME_CONTRACT_STATUS_Refused_MEAN', 'PREV_NAME_PRODUCT_TYPE_nan_MEAN', 'FLAG_DOCUMENT_14', 'CC_NAME_CONTRACT_STATUS_Refused_MIN',
    'CC_NAME_CONTRACT_STATUS_Refused_SUM', 'BURO_STATUS_nan_MEAN_MEAN', 'CC_NAME_CONTRACT_STATUS_Approved_MAX', 'OCCUPATION_TYPE_HR staff',
    'PREV_NAME_SELLER_INDUSTRY_Tourism_MEAN', 'PREV_NAME_PAYMENT_TYPE_nan_MEAN', 'FLAG_DOCUMENT_19', 'PREV_NAME_SELLER_INDUSTRY_nan_MEAN',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MIN', 'CC_SK_DPD_DEF_MIN', 'ORGANIZATION_TYPE_Culture', 'CC_NAME_CONTRACT_STATUS_Approved_MEAN',
    'NEW_RATIO_PREV_HUMAN_EVAL_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Gasification / water supply_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Hobby_MEAN', 'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MEAN', 'POS_NAME_CONTRACT_STATUS_Demand_MEAN',
    'FLAG_DOCUMENT_20', 'PREV_NAME_GOODS_CATEGORY_Animals_MEAN', 'PREV_NAME_GOODS_CATEGORY_Additional Service_MEAN',
    'PREV_NAME_CONTRACT_TYPE_nan_MEAN', 'PREV_NAME_CONTRACT_TYPE_XNA_MEAN', 'CC_NAME_CONTRACT_STATUS_Completed_MIN',
    'PREV_NAME_CONTRACT_STATUS_nan_MEAN', 'POS_NAME_CONTRACT_STATUS_XNA_MEAN', 'FLAG_DOCUMENT_2', 'CC_NAME_CONTRACT_STATUS_Approved_MIN',
    'FLAG_DOCUMENT_4', 'CC_NAME_CONTRACT_STATUS_Approved_SUM', 'PREV_CODE_REJECT_REASON_SYSTEM_MEAN', 'CLOSED_CREDIT_DAY_OVERDUE_MAX',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a home_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a garage_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Business development_MEAN', 'NEW_RATIO_PREV_HUMAN_EVAL_MAX', 'CLOSED_CREDIT_DAY_OVERDUE_MEAN',
    'PREV_FLAG_LAST_APPL_PER_CONTRACT_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_Active_MAX', 'CC_NAME_CONTRACT_STATUS_Approved_VAR',
    'PREV_CHANNEL_TYPE_Car dealer_MEAN', 'PREV_CODE_REJECT_REASON_nan_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MIN', 'FLAG_MOBIL',
    'HOUSETYPE_MODE_terraced house', 'ORGANIZATION_TYPE_Emergency', 'PREV_NAME_CLIENT_TYPE_nan_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_nan_MEAN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_SUM', 'ORGANIZATION_TYPE_Insurance',
    'NAME_EDUCATION_TYPE_Academic degree', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MEAN',
    'ORGANIZATION_TYPE_Legal Services', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX', 'NAME_FAMILY_STATUS_Unknown',
    'REFUSED_HUMAN_EVAL_MAX', 'OCCUPATION_TYPE_Realty agents', 'REFUSED_HUMAN_EVAL_MEAN', 'CC_NAME_CONTRACT_STATUS_Signed_MIN',
    'CC_SK_DPD_MIN', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Advertising', 'APPROVED_HUMAN_EVAL_MIN', 'FLAG_DOCUMENT_7',
    'ORGANIZATION_TYPE_Industry: type 8', 'APPROVED_HUMAN_EVAL_MEAN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MIN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Wedding / gift / holiday_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Refusal to name the goal_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Purchase of electronic equipment_MEAN', 'NEW_RATIO_PREV_HUMAN_EVAL_MIN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Money for a third person_MEAN', 'POS_NAME_CONTRACT_STATUS_nan_MEAN',
    'BURO_CREDIT_TYPE_Unknown type of loan_MEAN', 'ORGANIZATION_TYPE_Transport: type 1', 'PREV_NAME_YIELD_GROUP_nan_MEAN',
    'ORGANIZATION_TYPE_Industry: type 5', 'CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_VAR',
    'ORGANIZATION_TYPE_Industry: type 6', 'APPROVED_HUMAN_EVAL_MAX', 'BURO_CREDIT_TYPE_Real estate loan_MEAN',
    'CC_NAME_CONTRACT_STATUS_Demand_MAX', 'FLAG_DOCUMENT_13', 'BURO_CREDIT_TYPE_nan_MEAN', 'ORGANIZATION_TYPE_XNA',
    'CC_NAME_CONTRACT_STATUS_Demand_MEAN', 'CC_NAME_CONTRACT_STATUS_Demand_MIN', 'PREV_NAME_GOODS_CATEGORY_Weapon_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Tourism_MEAN', 'PREV_NAME_GOODS_CATEGORY_nan_MEAN', 'NEW_DOC_IND_AVG', 'CC_NAME_CONTRACT_STATUS_Demand_SUM',
    'CC_NAME_CONTRACT_STATUS_Demand_VAR', 'PREV_NAME_PAYMENT_TYPE_Cashless from the account of the employer_MEAN',
    'PREV_PRODUCT_COMBINATION_POS others without interest_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Furniture_MEAN',
    'BURO_CREDIT_TYPE_Loan for working capital replenishment_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Journey_MEAN',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'PREV_NAME_GOODS_CATEGORY_Medical Supplies_MEAN', 'FLAG_DOCUMENT_16', 'NAME_TYPE_SUITE_Other_A',
    'NAME_HOUSING_TYPE_Co-op apartment', 'PREV_NAME_CASH_LOAN_PURPOSE_Everyday expenses_MEAN', 'PREV_NAME_GOODS_CATEGORY_Other_MEAN',
    'PREV_NAME_CLIENT_TYPE_XNA_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_MIN', 'BURO_CREDIT_TYPE_Another type of loan_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Office Appliances_MEAN', 'ORGANIZATION_TYPE_Agriculture',
    'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN', 'CC_SK_DPD_DEF_MAX', 'PREV_NAME_GOODS_CATEGORY_Homewares_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_Education_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Payments on other loans_MEAN', 'FLAG_DOCUMENT_5',
    'ORGANIZATION_TYPE_Industry: type 11', 'FLAG_DOCUMENT_9', 'PREV_CODE_REJECT_REASON_XNA_MEAN', 'WALLSMATERIAL_MODE_Wooden',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_SUM', 'EMERGENCYSTATE_MODE_Yes', 'CC_NAME_CONTRACT_STATUS_Signed_MAX', 'ORGANIZATION_TYPE_University',
    'ORGANIZATION_TYPE_Housing', 'ACTIVE_CREDIT_DAY_OVERDUE_MEAN', 'NEW_RATIO_BURO_MONTHS_BALANCE_MAX_MAX',
    'CC_NAME_CONTRACT_STATUS_Signed_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Jewelry_MEAN', 'CC_NAME_CONTRACT_STATUS_Completed_SUM',
    'PREV_NAME_TYPE_SUITE_Group of people_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Buying a used car_MEAN', 'ORGANIZATION_TYPE_Trade: type 2',
    'NEW_RATIO_BURO_CREDIT_DAY_OVERDUE_MAX', 'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM', 'CC_SK_DPD_DEF_SUM', 'BURO_STATUS_5_MEAN_MEAN',
    'OCCUPATION_TYPE_Secretaries', 'CC_NAME_CONTRACT_STATUS_Completed_MEAN', 'PREV_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',

    # 1 < importance / null_importance <= 2
    'CC_NAME_CONTRACT_STATUS_Completed_VAR', 'CC_NAME_CONTRACT_STATUS_Completed_MAX', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_OVERDUE_MEAN',
    'ORGANIZATION_TYPE_Services', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN', 'ORGANIZATION_TYPE_Business Entity Type 2',
    'OCCUPATION_TYPE_Cleaning staff', 'CC_CNT_INSTALMENT_MATURE_CUM_MIN', 'POS_NAME_CONTRACT_STATUS_Approved_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Medicine_MEAN', 'CC_AMT_INST_MIN_REGULARITY_MIN', 'FLAG_EMP_PHONE', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MAX',
    'PREV_NAME_TYPE_SUITE_Other_A_MEAN', 'BURO_CREDIT_DAY_OVERDUE_MEAN', 'PREV_NAME_PAYMENT_TYPE_Non-cash from your account_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Vehicles_MEAN', 'ORGANIZATION_TYPE_Electricity', 'BURO_CNT_CREDIT_PROLONG_SUM', 'PREV_HUMAN_EVAL_MAX',
    'CC_CNT_DRAWINGS_OTHER_CURRENT_VAR', 'BURO_CREDIT_TYPE_Loan for business development_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Other_MEAN',
    'BURO_STATUS_3_MEAN_MEAN', 'WALLSMATERIAL_MODE_Mixed', 'ORGANIZATION_TYPE_Industry: type 1', 'BURO_MONTHS_BALANCE_MAX_MAX',
    'CC_AMT_DRAWINGS_OTHER_CURRENT_MEAN', 'CC_CNT_DRAWINGS_OTHER_CURRENT_MEAN', 'BURO_STATUS_4_MEAN_MEAN',
    'PREV_NAME_CONTRACT_STATUS_Unused offer_MEAN', 'PREV_CODE_REJECT_REASON_VERIF_MEAN', 'OCCUPATION_TYPE_Cooking staff',
    'OCCUPATION_TYPE_Managers', 'CC_AMT_DRAWINGS_OTHER_CURRENT_SUM', 'CC_AMT_PAYMENT_TOTAL_CURRENT_MIN',

    # 2 < importance / null_importance <= 3
    'CC_MONTHS_BALANCE_MIN', 'ORGANIZATION_TYPE_Trade: type 7', 'NEW_RATIO_BURO_CNT_CREDIT_PROLONG_SUM', 'CC_CNT_DRAWINGS_POS_CURRENT_MIN',
    'CC_SK_DPD_MAX', 'OCCUPATION_TYPE_Security staff', 'ORGANIZATION_TYPE_Transport: type 2', 'CC_SK_DPD_MEAN',
    'ACTIVE_CREDIT_DAY_OVERDUE_MAX', 'ORGANIZATION_TYPE_Security', 'CLOSED_CNT_CREDIT_PROLONG_SUM', 'CLOSED_AMT_CREDIT_SUM_LIMIT_MEAN',
    'NAME_TYPE_SUITE_Family', 'EMERGENCYSTATE_MODE_No', 'BURO_STATUS_2_MEAN_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MIN',
    'CC_AMT_DRAWINGS_CURRENT_MIN', 'FONDKAPREMONT_MODE_not specified', 'WALLSMATERIAL_MODE_Block',

    # 3 < importance / null_importance <= 4
    'CC_AMT_DRAWINGS_POS_CURRENT_MIN', 'NONLIVINGAPARTMENTS_MODE', 'CC_CNT_INSTALMENT_MATURE_CUM_MAX',
    'PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN', 'FLAG_DOCUMENT_6', 'CC_MONTHS_BALANCE_MAX', 'WALLSMATERIAL_MODE_Monolithic',
    'BURO_CREDIT_DAY_OVERDUE_MAX', 'YEARS_BUILD_AVG', 'CLOSED_AMT_CREDIT_SUM_DEBT_SUM', 'PREV_FLAG_LAST_APPL_PER_CONTRACT_Y_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Industry_MEAN', 'CC_MONTHS_BALANCE_VAR', 'PREV_NAME_GOODS_CATEGORY_Jewelry_MEAN', 'ELEVATORS_MODE',
    'CLOSED_MONTHS_BALANCE_MAX_MAX', 'CC_NAME_CONTRACT_STATUS_Active_MEAN', 'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN',
    'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MIN', 'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN', 'CC_CNT_DRAWINGS_CURRENT_MIN',
    'CLOSED_AMT_CREDIT_SUM_DEBT_MEAN', 'PREV_NAME_GOODS_CATEGORY_Construction Materials_MEAN', 'CC_COUNT',
    'PREV_NAME_SELLER_INDUSTRY_Construction_MEAN', 'INSTAL_PAYMENT_PERC_MAX', 'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MAX',
    'CC_AMT_RECIVABLE_MIN', 'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN', 'CC_SK_DPD_DEF_VAR',
    'CC_NAME_CONTRACT_STATUS_Active_VAR', 'CC_AMT_BALANCE_SUM', 'POS_NAME_CONTRACT_STATUS_Signed_MEAN', 'FLAG_EMAIL',
    'CC_AMT_CREDIT_LIMIT_ACTUAL_MAX',

    # 4 < importance / null_importance <= 5
    'FLOORSMIN_AVG', 'PREV_NAME_TYPE_SUITE_Other_B_MEAN', 'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
    'PREV_NAME_TYPE_SUITE_Children_MEAN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_MAX', 'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN',
    'PREV_CHANNEL_TYPE_Regional / Local_MEAN', 'COMMONAREA_AVG', 'PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN',
    'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX', 'HOUSETYPE_MODE_block of flats', 'WEEKDAY_APPR_PROCESS_START_THURSDAY',
    'PREV_PRODUCT_COMBINATION_POS industry without interest_MEAN', 'NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_SUM', 'ELEVATORS_AVG',
    'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX', 'LIVINGAPARTMENTS_MODE', 'REG_REGION_NOT_LIVE_REGION',
    'CC_NAME_CONTRACT_STATUS_Signed_VAR', 'YEARS_BEGINEXPLUATATION_AVG', 'CC_CNT_DRAWINGS_POS_CURRENT_SUM',
    'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN', 'CLOSED_MONTHS_BALANCE_MIN_MIN', 'CC_MONTHS_BALANCE_MEAN',
    'CC_AMT_CREDIT_LIMIT_ACTUAL_VAR', 'LIVINGAPARTMENTS_AVG', 'POS_SK_DPD_MAX', 'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MIN',
    'FONDKAPREMONT_MODE_reg oper account', 'AMT_REQ_CREDIT_BUREAU_MON', 'PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN',

    # 5 < importance / null_importance <= 6
    'CC_NAME_CONTRACT_STATUS_Active_SUM', 'ORGANIZATION_TYPE_Industry: type 3', 'NAME_HOUSING_TYPE_With parents', 'COMMONAREA_MODE',
    'CC_AMT_RECEIVABLE_PRINCIPAL_MIN', 'CC_CNT_DRAWINGS_ATM_CURRENT_MAX', 'CC_CNT_INSTALMENT_MATURE_CUM_SUM', 'BASEMENTAREA_AVG',
    'NEW_RATIO_BURO_MONTHS_BALANCE_MIN_MIN', 'CC_AMT_DRAWINGS_POS_CURRENT_MEAN', 'NEW_RATIO_BURO_AMT_ANNUITY_MAX',
    'NAME_FAMILY_STATUS_Civil marriage', 'FLOORSMAX_MODE', 'CC_CNT_DRAWINGS_ATM_CURRENT_SUM', 'NONLIVINGAPARTMENTS_MEDI',
    'PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MEAN', 'REFUSED_AMT_DOWN_PAYMENT_MAX', 'BASEMENTAREA_MODE',
    'REFUSED_HOUR_APPR_PROCESS_START_MAX', 'NONLIVINGAPARTMENTS_AVG', 'OCCUPATION_TYPE_Private service staff',
    'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN', 'ACTIVE_CNT_CREDIT_PROLONG_SUM', 'NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MEAN',
    'REFUSED_RATE_DOWN_PAYMENT_MEAN', 'FONDKAPREMONT_MODE_org spec account', 'CLOSED_MONTHS_BALANCE_SIZE_SUM',
    'NEW_RATIO_PREV_APP_CREDIT_PERC_VAR', 'REFUSED_APP_CREDIT_PERC_MAX', 'ENTRANCES_AVG', 'REFUSED_AMT_ANNUITY_MAX',
    'CC_AMT_INST_MIN_REGULARITY_MEAN', 'APARTMENTS_AVG', 'ACTIVE_MONTHS_BALANCE_MIN_MIN', 'REFUSED_RATE_DOWN_PAYMENT_MIN',
    'NAME_TYPE_SUITE_Children', 'ACTIVE_MONTHS_BALANCE_SIZE_SUM', 'ORGANIZATION_TYPE_Business Entity Type 1', 'CC_SK_DPD_VAR',
    'LIVE_REGION_NOT_WORK_REGION', 'CC_AMT_DRAWINGS_POS_CURRENT_MAX', 'NEW_RATIO_BURO_AMT_ANNUITY_MEAN',
    'NEW_RATIO_PREV_AMT_GOODS_PRICE_MIN', 'CC_AMT_PAYMENT_CURRENT_MAX', 'CC_CNT_INSTALMENT_MATURE_CUM_MEAN', 'CC_AMT_PAYMENT_CURRENT_MIN',
    'PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN', 'CC_AMT_DRAWINGS_CURRENT_MAX', 'NEW_RATIO_PREV_AMT_ANNUITY_MIN',

    # 6 < importance / null_importance <= 10
    'PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN', 'NONLIVINGAREA_AVG', 'REFUSED_CNT_PAYMENT_SUM', 'NONLIVINGAREA_MODE',
    'PREV_NAME_TYPE_SUITE_Spouse, partner_MEAN', 'ACTIVE_MONTHS_BALANCE_MAX_MAX', 'NEW_RATIO_BURO_DAYS_CREDIT_VAR',
    'NAME_FAMILY_STATUS_Widow', 'PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN', 'CC_AMT_CREDIT_LIMIT_ACTUAL_SUM',
    'BURO_MONTHS_BALANCE_MIN_MIN', 'LANDAREA_AVG', 'LIVINGAREA_AVG', 'PREV_NAME_SELLER_INDUSTRY_Furniture_MEAN',
    'NEW_RATIO_PREV_AMT_CREDIT_MEAN', 'BURO_DAYS_CREDIT_ENDDATE_MIN', 'FLOORSMIN_MODE', 'YEARS_BUILD_MODE',
    'CC_CNT_INSTALMENT_MATURE_CUM_VAR', 'REFUSED_AMT_GOODS_PRICE_MAX', 'PREV_CODE_REJECT_REASON_LIMIT_MEAN',
    'CLOSED_DAYS_CREDIT_ENDDATE_MIN', 'PREV_HOUR_APPR_PROCESS_START_MIN', 'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MAX',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'CNT_FAM_MEMBERS', 'PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN', 'PREV_AMT_DOWN_PAYMENT_MIN',
    'CC_AMT_INST_MIN_REGULARITY_MAX', 'PREV_PRODUCT_COMBINATION_Cash_MEAN', 'BURO_STATUS_X_MEAN_MEAN', 'ENTRANCES_MODE',
    'REFUSED_APP_CREDIT_PERC_VAR', 'LIVINGAREA_MODE', 'BURO_STATUS_C_MEAN_MEAN', 'PREV_NAME_GOODS_CATEGORY_Computers_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Gardening_MEAN', 'REFUSED_AMT_CREDIT_MAX', 'NEW_INC_PER_CHLD', 'ACTIVE_AMT_ANNUITY_MAX',
    'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MEAN', 'REFUSED_AMT_GOODS_PRICE_MIN', 'PREV_NAME_PORTFOLIO_XNA_MEAN',
    'CC_AMT_DRAWINGS_POS_CURRENT_VAR', 'FLOORSMAX_AVG', 'NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN', 'NEW_LIVE_IND_SUM', 'LANDAREA_MODE',
    'NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MEAN', 'BURO_AMT_ANNUITY_MAX', 'NEW_RATIO_PREV_APP_CREDIT_PERC_MIN',
    'PREV_PRODUCT_COMBINATION_Card Street_MEAN', 'PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN', 'CC_CNT_DRAWINGS_CURRENT_SUM',
    'CLOSED_MONTHS_BALANCE_SIZE_MEAN', 'CLOSED_DAYS_CREDIT_MIN', 'CC_AMT_DRAWINGS_OTHER_CURRENT_VAR', 'OBS_30_CNT_SOCIAL_CIRCLE',
    'CLOSED_AMT_ANNUITY_MEAN', 'REFUSED_HOUR_APPR_PROCESS_START_MIN', 'BURO_DAYS_CREDIT_VAR', 'REFUSED_AMT_ANNUITY_MEAN',
    'NONLIVINGAREA_MEDI', 'NAME_TYPE_SUITE_Unaccompanied', 'PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN', 'BURO_MONTHS_BALANCE_SIZE_MEAN',
    'CLOSED_DAYS_CREDIT_MEAN', 'CLOSED_DAYS_CREDIT_ENDDATE_MEAN', 'APPROVED_RATE_DOWN_PAYMENT_MIN', 'BURO_DAYS_CREDIT_UPDATE_MEAN',
    'NAME_TYPE_SUITE_Spouse, partner', 'CC_AMT_DRAWINGS_ATM_CURRENT_MAX', 'NEW_RATIO_PREV_DAYS_DECISION_MEAN',
    'APPROVED_AMT_DOWN_PAYMENT_MIN', 'NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_SUM',
    'NEW_RATIO_PREV_AMT_CREDIT_MIN', 'CC_AMT_INST_MIN_REGULARITY_SUM', 'PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN',
    'ACTIVE_AMT_ANNUITY_MEAN', 'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_XAP_MEAN',
    'APPROVED_HOUR_APPR_PROCESS_START_MIN', 'PREV_NAME_TYPE_SUITE_Family_MEAN', 'PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN',
    'PREV_CODE_REJECT_REASON_HC_MEAN', 'NEW_RATIO_BURO_DAYS_CREDIT_UPDATE_MEAN', 'NAME_HOUSING_TYPE_Rented apartment',
    'CC_AMT_PAYMENT_CURRENT_VAR', 'PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN', 'REFUSED_DAYS_DECISION_MIN', 'YEARS_BUILD_MEDI',
    'FLAG_OWN_REALTY', 'REFUSED_AMT_CREDIT_MEAN', 'NEW_RATIO_PREV_AMT_ANNUITY_MEAN', 'CC_MONTHS_BALANCE_SUM',
    'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN', 'PREV_CNT_PAYMENT_SUM', 'PREV_AMT_CREDIT_MIN', 'ACTIVE_DAYS_CREDIT_MIN', 'TOTALAREA_MODE',
    'REFUSED_RATE_DOWN_PAYMENT_MAX', 'BURO_CREDIT_ACTIVE_Sold_MEAN', 'EXT_SOURCE_1_VAR', 'REFUSED_AMT_APPLICATION_MEAN',
    'ACTIVE_DAYS_CREDIT_VAR', 'POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN', 'BURO_AMT_ANNUITY_MEAN',
    'NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'NEW_RATIO_BURO_DAYS_CREDIT_MIN', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM',
    'NEW_RATIO_PREV_AMT_APPLICATION_MAX', 'CLOSED_AMT_ANNUITY_MAX', 'PREV_AMT_APPLICATION_MIN', 'PREV_AMT_APPLICATION_MAX',
    'NEW_RATIO_PREV_APP_CREDIT_PERC_MAX', 'ORGANIZATION_TYPE_Other', 'PREV_AMT_GOODS_PRICE_MIN', 'BURO_DAYS_CREDIT_MIN', 'ELEVATORS_MEDI',
    'ACTIVE_MONTHS_BALANCE_SIZE_MEAN', 'PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN', 'CC_SK_DPD_DEF_MEAN', 'BURO_STATUS_1_MEAN_MEAN',
    'PREV_CODE_REJECT_REASON_SCOFR_MEAN', 'CNT_CHILDREN', 'PREV_NAME_PRODUCT_TYPE_x-sell_MEAN',
    'PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN', 'CC_AMT_DRAWINGS_ATM_CURRENT_VAR', 'PREV_NAME_CONTRACT_STATUS_Canceled_MEAN',
    'PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN', 'PREV_RATE_DOWN_PAYMENT_MIN', 'PREV_AMT_GOODS_PRICE_MEAN',
    'NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_SUM', 'CC_AMT_CREDIT_LIMIT_ACTUAL_MIN',
    'NEW_RATIO_PREV_AMT_ANNUITY_MAX', 'CLOSED_DAYS_CREDIT_UPDATE_MEAN', 'BASEMENTAREA_MEDI', 'APPROVED_APP_CREDIT_PERC_VAR',
    'CLOSED_AMT_CREDIT_SUM_DEBT_MAX', 'CC_AMT_RECEIVABLE_PRINCIPAL_SUM', 'APARTMENTS_MODE', 'CC_AMT_PAYMENT_TOTAL_CURRENT_VAR',
    'ORGANIZATION_TYPE_Government', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'PREV_CHANNEL_TYPE_Contact center_MEAN', 'FLOORSMAX_MEDI',
    'PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN', 'REFUSED_AMT_APPLICATION_MIN', 'PREV_CHANNEL_TYPE_Stone_MEAN',
    'NEW_RATIO_BURO_AMT_CREDIT_SUM_MEAN', 'REG_REGION_NOT_WORK_REGION', 'INCOME_PER_PERSON', 'INSTAL_AMT_INSTALMENT_MEAN',
    'CC_AMT_PAYMENT_TOTAL_CURRENT_MAX', 'CLOSED_DAYS_CREDIT_VAR', 'AMT_INCOME_TOTAL', 'REFUSED_CNT_PAYMENT_MEAN', 'LIVINGAPARTMENTS_MEDI',
    'NAME_FAMILY_STATUS_Separated', 'REFUSED_AMT_APPLICATION_MAX', 'APPROVED_AMT_CREDIT_MEAN', 'POS_NAME_CONTRACT_STATUS_Completed_MEAN',
    'INSTAL_PAYMENT_DIFF_VAR', 'PREV_HUMAN_EVAL_MIN', 'ORGANIZATION_TYPE_Transport: type 4', 'PREV_NAME_CLIENT_TYPE_Repeater_MEAN',
    'PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN', 'NEW_RATIO_PREV_AMT_APPLICATION_MIN', 'BURO_DAYS_CREDIT_ENDDATE_MEAN',
    'APPROVED_AMT_CREDIT_MIN', 'PREV_HOUR_APPR_PROCESS_START_MEAN', 'APPROVED_HOUR_APPR_PROCESS_START_MEAN', 'PREV_DAYS_DECISION_MIN',
    'REG_CITY_NOT_WORK_CITY', 'NEW_LIVE_IND_KURT', 'NEW_RATIO_PREV_CNT_PAYMENT_MEAN', 'OCCUPATION_TYPE_Waiters/barmen staff',
    'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN', 'NEW_RATIO_BURO_DAYS_CREDIT_MEAN', 'REFUSED_AMT_ANNUITY_MIN',
    'CC_AMT_DRAWINGS_CURRENT_VAR', 'POS_SK_DPD_MEAN', 'PREV_AMT_APPLICATION_MEAN', 'REFUSED_HOUR_APPR_PROCESS_START_MEAN',
    'NEW_RATIO_PREV_AMT_CREDIT_MAX', 'BURO_AMT_CREDIT_SUM_LIMIT_SUM', 'PREV_APP_CREDIT_PERC_VAR', 'ORGANIZATION_TYPE_Trade: type 3',
    'PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN', 'HOUR_APPR_PROCESS_START', 'NEW_RATIO_PREV_AMT_APPLICATION_MEAN',
    'PREV_AMT_ANNUITY_MAX', 'PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN', 'CC_AMT_DRAWINGS_CURRENT_MEAN',
    'PREV_AMT_CREDIT_MEAN', 'NEW_RATIO_PREV_AMT_GOODS_PRICE_MAX', 'NEW_RATIO_PREV_AMT_GOODS_PRICE_MEAN', 'APPROVED_AMT_APPLICATION_MEAN',
    'REFUSED_AMT_GOODS_PRICE_MEAN', 'NEW_RATIO_BURO_AMT_CREDIT_SUM_MAX', 'NEW_RATIO_BURO_DAYS_CREDIT_MAX',
    'NEW_RATIO_PREV_DAYS_DECISION_MIN', 'PREV_AMT_CREDIT_MAX', 'PREV_DAYS_DECISION_MAX', 'PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN',
    'NEW_CREDIT_TO_INCOME_RATIO', 'BURO_CREDIT_TYPE_Consumer credit_MEAN', 'INSTAL_COUNT', 'PREV_DAYS_DECISION_MEAN',
    'INSTAL_PAYMENT_PERC_VAR', 'PREV_NAME_PORTFOLIO_Cash_MEAN', 'CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN',

    # 10 < importance / null_importance <= 12
    'CLOSED_AMT_CREDIT_SUM_MAX', 'APPROVED_RATE_DOWN_PAYMENT_MEAN', 'WEEKDAY_APPR_PROCESS_START_SATURDAY',
    'PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN', 'PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN', 'INSTAL_AMT_PAYMENT_MIN',
    'BURO_STATUS_0_MEAN_MEAN', 'CC_AMT_DRAWINGS_POS_CURRENT_SUM', 'FONDKAPREMONT_MODE_reg oper spec account',
    'BURO_MONTHS_BALANCE_SIZE_SUM', 'REFUSED_APP_CREDIT_PERC_MEAN', 'REFUSED_DAYS_DECISION_MAX', 'PREV_TIME_DECAYED_HUMAN_EVAL_SUM',
    'ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'PREV_NAME_YIELD_GROUP_middle_MEAN', 'CC_AMT_BALANCE_MIN', 'APPROVED_AMT_GOODS_PRICE_MEAN',
    'PREV_AMT_ANNUITY_MIN', 'PREV_NAME_YIELD_GROUP_low_normal_MEAN', 'APPROVED_AMT_DOWN_PAYMENT_MEAN', 'CC_CNT_DRAWINGS_POS_CURRENT_MEAN',
    'CC_AMT_DRAWINGS_CURRENT_SUM', 'PREV_NAME_SELLER_INDUSTRY_XNA_MEAN', 'APPROVED_AMT_APPLICATION_MIN', 'APPROVED_AMT_ANNUITY_MIN',
    'AMT_REQ_CREDIT_BUREAU_DAY', 'APPROVED_HOUR_APPR_PROCESS_START_MAX', 'PREV_CHANNEL_TYPE_Country-wide_MEAN',
    'PREV_RATE_DOWN_PAYMENT_MAX', 'PREV_HOUR_APPR_PROCESS_START_MAX', 'PREV_NAME_PRODUCT_TYPE_XNA_MEAN',
    'PREV_NAME_GOODS_CATEGORY_Mobile_MEAN', 'CC_AMT_BALANCE_VAR', 'APPROVED_AMT_ANNUITY_MAX', 'INSTAL_DAYS_ENTRY_PAYMENT_SUM',
    'PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN', 'PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN',
    'ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN', 'APPROVED_AMT_GOODS_PRICE_MIN', 'CLOSED_AMT_CREDIT_SUM_SUM', 'CLOSED_AMT_CREDIT_SUM_MEAN',
    'APPROVED_APP_CREDIT_PERC_MEAN', 'CC_CNT_DRAWINGS_POS_CURRENT_VAR', 'APPROVED_DAYS_DECISION_MIN', 'YEARS_BEGINEXPLUATATION_MODE',
    'BURO_AMT_CREDIT_SUM_DEBT_MAX', 'COMMONAREA_MEDI', 'NAME_INCOME_TYPE_Commercial associate', 'PREV_AMT_DOWN_PAYMENT_MEAN',
    'APPROVED_RATE_DOWN_PAYMENT_MAX', 'NAME_EDUCATION_TYPE_Incomplete higher', 'APPROVED_APP_CREDIT_PERC_MAX', 'DAYS_REGISTRATION',
    'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_AMT_INSTALMENT_MAX', 'APPROVED_DAYS_DECISION_MEAN', 'PREV_AMT_GOODS_PRICE_MAX',
    'INSTAL_DAYS_ENTRY_PAYMENT_MEAN', 'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX',
    'BURO_CREDIT_ACTIVE_Active_MEAN', 'WEEKDAY_APPR_PROCESS_START_FRIDAY', 'NEW_INC_BY_ORG',
    'PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN', 'CC_SK_DPD_SUM', 'REFUSED_DAYS_DECISION_MEAN', 
    'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN', 'FLOORSMIN_MEDI', 'PREV_AMT_DOWN_PAYMENT_MAX', 'CC_AMT_RECEIVABLE_PRINCIPAL_VAR',
    'BURO_DAYS_CREDIT_MEAN', 'POS_MONTHS_BALANCE_MEAN', 'PREV_NAME_GOODS_CATEGORY_Furniture_MEAN', 'LANDAREA_MEDI',
    'NEW_RATIO_PREV_DAYS_DECISION_MAX', 'CC_AMT_PAYMENT_CURRENT_MEAN', 'HOUSETYPE_MODE_specific housing', 'INSTAL_DBD_MEAN',
]

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

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

def group_target_by_cols(df, recipe, residual = False):
    for m in range(len(recipe)):
        cols = recipe[m][0]
        for n in range(len(recipe[m][1])):
            target = recipe[m][1][n][0]
            method = recipe[m][1][n][1]
            name_grouped_target = target+'_BY_'+'_'.join(cols)
            tmp = df[cols + [target]].groupby(cols).agg(method)
            tmp = tmp.reset_index().rename(index=str, columns={target: name_grouped_target})
            df = df.merge(tmp, how='left', on=cols)
            if residual: df[name_grouped_target] = df[target] - df[name_grouped_target]
    return df

# Preprocess application_train.csv and application_test.csv
def application_train_test(nan_as_category = False):
    # Read data and merge
    df = read_df('application_train')
    test_df = read_df('application_test')
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f)]

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NUM_INSTALMENTS'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] 
    df['DIFF_CREDIT_AND_GOODS_RATIO'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_STD'] = df[docs].std(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)

    recipe = [
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [
            ('AMT_INCOME_TOTAL', 'median'),
        ]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [
            ('AMT_CREDIT', 'mean'),
            ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
            ('APARTMENTS_AVG', 'mean'),
            ('BASEMENTAREA_AVG', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('EXT_SOURCE_2', 'mean'),
            ('EXT_SOURCE_3', 'mean'),
            ('NONLIVINGAREA_AVG', 'mean'),
            ('OWN_CAR_AGE', 'mean'),
            ('YEARS_BUILD_AVG', 'mean'),
        ]),
        (['OCCUPATION_TYPE'], [
            ('AMT_CREDIT', 'mean'),
            ('CNT_CHILDREN', 'mean'),
            ('CNT_FAM_MEMBERS', 'mean'),
            ('DAYS_BIRTH', 'mean'),
            ('DAYS_EMPLOYED', 'mean'),
            ('DAYS_ID_PUBLISH', 'mean'),
            ('DAYS_REGISTRATION', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('EXT_SOURCE_2', 'mean'),
            ('EXT_SOURCE_3', 'mean'),
        ]),
        (['CODE_GENDER', 'OCCUPATION_TYPE'], [
            ('AMT_ANNUITY', 'mean'),
            ('CNT_CHILDREN', 'mean'),
            ('CNT_FAM_MEMBERS', 'mean'),
            ('DAYS_BIRTH', 'mean'),
            ('DAYS_EMPLOYED', 'mean'),
            ('DAYS_ID_PUBLISH', 'mean'),
            ('DAYS_REGISTRATION', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('EXT_SOURCE_2', 'mean'),
            ('EXT_SOURCE_3', 'mean'),
        ]),
        (['REG_CITY_NOT_WORK_CITY', 'OCCUPATION_TYPE'], [
            ('AMT_ANNUITY', 'mean'),
            ('CNT_CHILDREN', 'mean'),
            ('CNT_FAM_MEMBERS', 'mean'),
            ('DAYS_BIRTH', 'mean'),
            ('DAYS_EMPLOYED', 'mean'),
            ('DAYS_ID_PUBLISH', 'mean'),
            ('DAYS_REGISTRATION', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('EXT_SOURCE_2', 'mean'),
            ('EXT_SOURCE_3', 'mean'),
        ]),
        (['NAME_INCOME_TYPE', 'OCCUPATION_TYPE'], [
            ('AMT_ANNUITY', 'mean'),
            ('DAYS_BIRTH', 'mean'),
            ('DAYS_EMPLOYED', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('AMT_CREDIT', 'mean'),
            ('AMT_GOODS_PRICE', 'mean'),
            ('AMT_INCOME_TOTAL', 'mean'),
        ]),
        (['ORGANIZATION_TYPE', 'OCCUPATION_TYPE'], [
            ('AMT_ANNUITY', 'mean'),
            ('CNT_CHILDREN', 'mean'),
            ('CNT_FAM_MEMBERS', 'mean'),
            ('DAYS_BIRTH', 'mean'),
            ('DAYS_EMPLOYED', 'mean'),
            ('DAYS_ID_PUBLISH', 'mean'),
            ('DAYS_REGISTRATION', 'mean'),
            ('EXT_SOURCE_1', 'mean'),
            ('EXT_SOURCE_2', 'mean'),
            ('EXT_SOURCE_3', 'mean'),
        ]),
    ]
    # to do: change all group_target_by_cols to residual_group_target_by_cols?
    df = group_target_by_cols(df, recipe, residual = RESIDUAl_AGG)
    # df = group_target_by_cols(df, target = 'AMT_INCOME_TOTAL', cols = ['CODE_GENDER', 'NAME_EDUCATION_TYPE'], method='median')

    # df = group_target_by_cols(df, target = 'AMT_CREDIT', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean') 
    # df = group_target_by_cols(df, target = 'AMT_REQ_CREDIT_BUREAU_YEAR', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'APARTMENTS_AVG', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'BASEMENTAREA_AVG', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'NONLIVINGAREA_AVG', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'OWN_CAR_AGE', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'YEARS_BUILD_AVG', cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], method='mean')

    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['OCCUPATION_TYPE'], method='mean')

    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['CODE_GENDER','OCCUPATION_TYPE'], method='mean')



    # # fe24 slightly good -> to do: some are good?
    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['FLAG_EMP_PHONE','OCCUPATION_TYPE'], method='mean')

    # # fe25 slightly good -> to do: some are good?
    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['FLAG_OWN_CAR','OCCUPATION_TYPE'], method='mean')

    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['REG_CITY_NOT_WORK_CITY','OCCUPATION_TYPE'], method='mean')

    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'AMT_CREDIT', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'AMT_GOODS_PRICE', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'AMT_INCOME_TOTAL', cols = ['NAME_INCOME_TYPE','OCCUPATION_TYPE'], method='mean')

    # df = group_target_by_cols(df, target = 'AMT_ANNUITY', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_CHILDREN', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'CNT_FAM_MEMBERS', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_BIRTH', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_EMPLOYED', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_ID_PUBLISH', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'DAYS_REGISTRATION', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_1', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_2', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')
    # df = group_target_by_cols(df, target = 'EXT_SOURCE_3', cols = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'], method='mean')


    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['INCOME_TO_GOODS_PRICE_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_GOODS_PRICE']
    df['DIFF_INCOME_AND_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']

    # replace NEW_SCORES_STD
    df['EXT_SOURCE_1_VAR'] = (df['EXT_SOURCE_1'] - df['NEW_EXT_SOURCES_MEAN'])**2
    df['EXT_SOURCE_2_VAR'] = (df['EXT_SOURCE_2'] - df['NEW_EXT_SOURCES_MEAN'])**2
    df['EXT_SOURCE_3_VAR'] = (df['EXT_SOURCE_3'] - df['NEW_EXT_SOURCES_MEAN'])**2
    df['EXT_SOURCE_1_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_1_VAR'].median())
    df['EXT_SOURCE_2_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_2_VAR'].median())
    df['EXT_SOURCE_3_VAR'] = df['EXT_SOURCE_1_VAR'].fillna(df['EXT_SOURCE_3_VAR'].median())

    df['SOCIAL_PRED'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['DEF_30_CNT_SOCIAL_CIRCLE'] + 2*df['OBS_60_CNT_SOCIAL_CIRCLE'] + 2*df['DEF_60_CNT_SOCIAL_CIRCLE']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # From Aguiar: Some simple new features (percentages)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(nan_as_category = True):
    bureau = read_df('bureau')
    bb = read_df('bureau_balance')
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = active_agg.columns.tolist()
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    
    for e in cols:
        bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]
    
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(nan_as_category = True):
    prev = read_df('previous_application')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['HUMAN_EVAL'] = prev['NAME_CONTRACT_STATUS_Approved'] - prev['NAME_CONTRACT_STATUS_Refused']
    prev['TIME_DECAYED_HUMAN_EVAL'] = prev['HUMAN_EVAL'] / prev['DAYS_DECISION']**2

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'HUMAN_EVAL': ['min', 'max', 'mean', 'sum'],
        'TIME_DECAYED_HUMAN_EVAL': ['sum'],
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    cols = approved_agg.columns.tolist()
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    
    for e in cols:
        prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]
    
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(nan_as_category = True):
    pos = read_df('POS_CASH_balance')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    pos['TIME_DECAYED_UNPAYED_RATIO'] = pos['CNT_INSTALMENT_FUTURE'] / pos['CNT_INSTALMENT'] / pos['MONTHS_BALANCE']**2

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'TIME_DECAYED_UNPAYED_RATIO': ['max', 'mean', 'size', 'sum'],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean',]
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(df, nan_as_category = True):
    ins = read_df('installments_payments')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'] # rm1 
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT'] # rm2
    # Days past due and days before due (no negative values)
    ins['DPD'] = (ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']) / ins['DAYS_INSTALMENT']**2
    ins['DBD'] = (ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']) / ins['DAYS_INSTALMENT']**2
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # # Features: Perform aggregations
    # aggregations1 = {
    #     'NUM_INSTALMENT_VERSION': ['nunique'],
    #     'DPD': ['max', 'mean', 'sum'],
    #     'DBD': ['max', 'mean', 'sum'],
    #     'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'], # rm1
    #     'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'], # rm2
    #     'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    #     'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    #     'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    # }
    # for cat in cat_cols:
    #     aggregations1[cat] = ['mean']
    # ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations1)
    # ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # ============================================== fe2 ok ==============================================
    aggregations1 = {
        'SK_ID_CURR': ['first'],
        'NUM_INSTALMENT_VERSION': ['max', 'size'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'sum'],
        'PAYMENT_PERC': ['mean', 'sum'],
        'PAYMENT_DIFF': ['max', 'sum'],
        'AMT_INSTALMENT': ['sum'],
        'AMT_PAYMENT': ['min', 'max', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max'],
        'DAYS_INSTALMENT': ['max', 'min'],
        # 'NUM_INSTALMENT_NUMBER': ['max'], # fe5
    }
    for cat in cat_cols:
        aggregations1[cat] = ['mean']
    ins_agg = ins.groupby(['SK_ID_PREV']).agg(aggregations1)
    ins_agg.columns = pd.Index([e[0] if e[0]=='SK_ID_CURR' else (e[0]+'_'+e[1].upper()) for e in ins_agg.columns.tolist()])

    timestamp = (ins_agg['DAYS_INSTALMENT_MAX'] + ins_agg['DAYS_INSTALMENT_MIN']) / 2
    # ins_agg['NUM_INSTALMENT_VERSION_MAX'] = ins_agg['NUM_INSTALMENT_VERSION_MAX'] / timestamp**2 # fe3.1
    # ins_agg['NUM_INSTALMENT_VERSION_SIZE'] = ins_agg['NUM_INSTALMENT_VERSION_SIZE'] / timestamp**2 # fe4.1
    # ins_agg['VERSIONS_CHANGES_RATIO'] = ins_agg['NUM_INSTALMENT_VERSION_MAX'] / ins_agg['NUM_INSTALMENT_VERSION_SIZE'] # fe6
    # ins_agg['VERSIONS_CHANGES_RATIO'] = ins_agg['NUM_INSTALMENT_VERSION_MAX'] / ins_agg['NUM_INSTALMENT_VERSION_SIZE'] / timestamp**2 # fe6.1
    # ins_agg['VERSIONS_CHANGES_DIFF'] = ins_agg['NUM_INSTALMENT_VERSION_MAX'] - ins_agg['NUM_INSTALMENT_VERSION_SIZE'] # fe7
    # ins_agg['VERSIONS_CHANGES_DIFF'] = (ins_agg['NUM_INSTALMENT_VERSION_MAX'] - ins_agg['NUM_INSTALMENT_VERSION_SIZE']) / timestamp**2 # fe7.1
    # to do: compare "term" between past and now
    aggregations2 = {
        # 'NUM_INSTALMENT_VERSION_MAX': ['mean'], # fe3, 3.1
        # 'NUM_INSTALMENT_VERSION_SIZE': ['mean'], # fe4, 4.1
        'DPD_MAX': ['max'],
        'DPD_MEAN': ['mean'],
        'DPD_SUM': ['sum'],
        'DBD_MAX': ['max'],
        'DBD_SUM': ['sum'],
        'PAYMENT_PERC_MEAN': ['mean'],        
        'PAYMENT_PERC_SUM': ['sum'],
        'PAYMENT_DIFF_MAX': ['max'],
        'PAYMENT_DIFF_SUM': ['sum'],
        'AMT_INSTALMENT_SUM': ['sum'],
        'AMT_PAYMENT_MIN': ['min'],
        'AMT_PAYMENT_MAX': ['max'],
        'AMT_PAYMENT_SUM': ['sum'],
        'DAYS_ENTRY_PAYMENT_MAX': ['max'],
        # 'NUM_INSTALMENT_NUMBER_MAX': ['max', 'min', 'median'], # fe5
        # 'VERSIONS_CHANGES_RATIO': ['min', 'max', 'median', 'var'], # fe6, 6.1
        # 'VERSIONS_CHANGES_DIFF': ['min', 'max', 'median', 'var'], # fe7, 7.1
    }
    ins_agg = ins_agg.groupby('SK_ID_CURR').agg(aggregations2)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + '_' + e[1].upper() for e in ins_agg.columns.tolist()])
    # =======================================================================================================

    # # Count installments accounts
    # ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(nan_as_category = True):
    cc = read_df('credit_card_balance')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)

    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=int(time.time()))
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=int(time.time()))
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

def main():
    df = pd.DataFrame()
    with timer("Process bureau and bureau_balance"):
        df = application_train_test()
        print("Train/Test application df shape:", df.shape)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance()
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications()
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash()
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(df)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance()
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        print(df.shape)
        df.drop(feature_graveyard, axis=1, inplace=True, errors='ignore')
        gc.collect()
        print(df.shape)   

        feature_importance_df = pd.DataFrame()
        over_folds_val_auc_list = np.zeros(ITERATION)
        for i in range(ITERATION):
            print('Iteration %i' %i)
            iter_feat_imp, over_folds_val_auc = kfold_lightgbm(df, num_folds= NUM_FOLDS, stratified= STRATIFIED)
            feature_importance_df = pd.concat([feature_importance_df, iter_feat_imp], axis=0)
            over_folds_val_auc_list[i] = over_folds_val_auc

        print('============================================\nOver-iterations val AUC score %.6f' %over_folds_val_auc_list.mean())
        print('Standard deviation %.6f\n============================================' %over_folds_val_auc_list.std())
        # display_importances(feature_importance_df)
        feature_importance_df_median = feature_importance_df[["feature", "importance"]].groupby("feature").median().sort_values(by="importance", ascending=False)
        useless_features_df = feature_importance_df_median.loc[feature_importance_df_median['importance'] == 0]
        feature_importance_df_mean = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)

        if TEST_NULL_HYPO:
            feature_importance_df_mean.to_csv("feature_importance-null_hypo.csv", index = True)
        else:
            feature_importance_df_mean.to_csv("feature_importance.csv", index = True)
            useless_features_list = useless_features_df.index.tolist()
            print('useless/overfitting features: \'' + '\', \''.join(useless_features_list) + '\'')

if __name__ == "__main__":
    submission_file_name = "sub.csv"
    with timer("Full model run"):
        main()