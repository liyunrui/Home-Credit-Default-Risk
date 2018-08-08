# -*- coding:utf8
import pandas as pd
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_df(name):
    if os.path.exists('../input/%s.h5' %name):
        df = pd.read_hdf('../input/%s.h5' %name, str(name))
    else:
        df = pd.read_csv('../input/%s.csv' %name)
        df.to_hdf('../input/%s.h5' %name, str(name))
    return df

#-----------------------------------
# Read data and merge
#-----------------------------------
df = read_df('application_train')
test_df = read_df('application_test')
print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
df = df.append(test_df).reset_index()

#-----------------------------------
# simple processing
#-----------------------------------

# Optional: Remove 4 applications with XNA CODE_GENDER (train set)
df = df[df['CODE_GENDER'] != 'XNA']
# NaN values for DAYS_EMPLOYED: 365.243 -> nan
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

#-----------------------------------
# age
#-----------------------------------
df['age'] = df['DAYS_BIRTH'] / -365.0
df['is_age_betwee_20_and_25'] = df.age.apply(lambda x: 1.0 if 20 < x <= 25.0 else 0)
df['is_age_betwee_25_and_30'] = df.age.apply(lambda x: 1.0 if 25 < x <= 30.0 else 0)
df['is_age_betwee_30_and_35'] = df.age.apply(lambda x: 1.0 if 30 < x <= 35.0 else 0)
df['is_age_betwee_35_and_40'] = df.age.apply(lambda x: 1.0 if 35 < x <= 40.0 else 0)
df['is_age_betwee_40_and_45'] = df.age.apply(lambda x: 1.0 if 40 < x <= 45.0 else 0)
df['is_age_betwee_45_and_50'] = df.age.apply(lambda x: 1.0 if 45 < x <= 50.0 else 0)
df['is_age_above_50'] = df.age.apply(lambda x: 1.0 if x >= 50.0 else 0)


#-----------------------------------
# 在這次申請貸款之前幾年, 這個人開始現在的工作。如是0.5, 這個人這次申請這貸款前半年開始他現在這份工作。
#-----------------------------------
df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365.0
df['maybe_they_have_no_job_before_half_year'] = df.YEARS_EMPLOYED.apply(lambda x: 1 if x <=0.5 else 0)
df['maybe_they_have_no_job_before_1_or_half_year'] = df.YEARS_EMPLOYED.apply(lambda x: 1 if 0.5 <= x <=1.0 else 0)
df['they_must_do_other_thing_to_survive'] = df.YEARS_EMPLOYED.apply(lambda x: 1 if x > 1 else 0)

#-----------------------------------
# 大學剛畢業就有車有房還貸款, 肯定裝逼, 還不起。
#-----------------------------------
df['fresh_have_car'] = [1 if (age < 22) and (have_car == 'Y') else 0 for have_car, age in zip(df.FLAG_OWN_CAR, df.age)]
df['fresh_have_house'] = [1 if (age < 22) and (have_house == 'Y') else 0 for have_house, age in zip(df.FLAG_OWN_REALTY, df.age)]
df['fresh_have_house_and_car'] = [1 if (age < 22) and (have_house == 'Y') and (have_car == 'Y') else 0 for have_house,have_car,age in zip(df.FLAG_OWN_REALTY, df.FLAG_OWN_CAR,df.age)]
#-----------------------------------
# 老人沒車沒房還貸款, 肯定高危險份子,還不起。
#-----------------------------------

df['old_no_car'] = [1 if (age > 50) and (have_car == 'N') else 0 for have_car, age in zip(df.FLAG_OWN_CAR, df.age)]
df['old_no_house'] = [1 if (age > 50) and (have_house == 'N') else 0 for have_house, age in zip(df.FLAG_OWN_REALTY, df.age)]
df['old_no_house_and_car'] = [1 if (age > 50) and (have_house == 'N') and (have_car == 'N') else 0 for have_house,have_car,age in zip(df.FLAG_OWN_REALTY, df.FLAG_OWN_CAR,df.age)]


#-----------------------------------
#save 
#-----------------------------------
output = ['SK_ID_CURR'] \
+ [f for f in df.columns.tolist() if ('age' in f) or ('year') in f] \
+ ['YEARS_EMPLOYED','they_must_do_other_thing_to_survive'] \
+ [f for f in df.columns.tolist() if 'fresh' in f] \
+ [f for f in df.columns.tolist() if 'old_no' in f]

df[output].to_hdf('../features/date_feature.h5', 'date_feature')




