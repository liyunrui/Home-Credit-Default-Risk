'''
Created on July 27 2018

@author: Ray

加了一坨30個feature, 
兩種case:
Case1: 如果score比base_feature好, 我們就把這30坨進去base_fautres(base_feature augmentation)
Case2: 如果score比base_feature差, 我們重做feature selection 實驗。

'''

import pandas as pd
import numpy as np

def application_train_test(df):
	'''
	read features and merge different feature together.

	parameters:
	---------------------
	df: DataFrame
	name: train or test
	'''
	# age, car, house, if there is no job
	df = pd.merge(df, pd.read_hdf('../features/date_feature.h5', 'date_feature'), 
		on = ['SK_ID_CURR'],
		how = 'left'
		)
	return df

def concat_all_features():

	#--------------------
	# laod base features
	#--------------------	
	df = pd.read_hdf('../features/base_featurs.h5', 'base_featurs')
	# remove the raw features already used for creating new feature 
	already_used_raw_features = ['DAYS_EMPLOYED', 'DAYS_BIRTH']
	for a_u_r_features in already_used_raw_features:
		if a_u_r_features in df.columns.tolist():
			df.drop([a_u_r_features], axis = 1, inplace = True)
			print (df.shape)
	#--------------------
	# concat features
	#--------------------	
	df = application_train_test(df)

	#-------------------- 
	# feature engineering
	#--------------------
	no_need_to_log_transform = ['SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']
	# some features with largest, we perform log transformation to them
	for col in df.columns:
		if col not in no_need_to_log_transform:
			if (df[col].dtypes == np.int64) and (df[col].max() > 100) and (df[col].min() > 0):
				print ('col',col)
				df['log_{}'.format(col)] = np.log(df[col] + 1) # smoothing
				df.drop(col, axis = 1, inplace = True)
			elif (df[col].dtypes == np.float64) and (df[col].max() > 100) and (df[col].min() > 0):
				print ('col',col)
				df['log_{}'.format(col)] = np.log(df[col] + 1) # smoothing
				df.drop(col, axis = 1, inplace = True)
			else:
				pass
	#--------------------
	# remove features we don't need to feed into the tree.
	#--------------------
	df.drop(['index'], axis = 1, inplace = True)
	#--------------------
	# output
	#--------------------
	print (df.shape)
	df.to_hdf('../features/all_features.h5', 'all_features')
	# logging
	print ('output all features')

if __name__ == '__main__':
	##################################################
	# Main
	##################################################
	concat_all_features()