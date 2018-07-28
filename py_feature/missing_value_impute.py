from ycimpute.imputer.knnimput import KNN # inputation library
import pandas as pd
import numpy as np
import time
import os
import multiprocessing as mp # for speeding up some process

df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
no_need_to_inpute = ['SK_ID_CURR','index']
df.drop(no_need_to_inpute,axis = 1, inplace = True)
print (df.shape)

#---------------------------------------------
# Step1: split dataframe into multiple dataframe to avoid memory problem
#---------------------------------------------

small_df_ls = []
# shuffle
df = df.sample(frac = 1.0)
print ('Big DataFrame',df.shape)
num_split = 25
previous_step = None
for i, step in enumerate(np.arange(0, df.shape[0], step = int(df.shape[0] / num_split))):
    # for memory problem, we cannot feed all the data points into algorithm, which depends on row and n_features
    if i == 0:
        pass
    elif i == 1:
        small_df_ls.append(df[0: step])
        previous_step = step
    elif i == (num_split):
        small_df_ls.append(df[previous_step: ])
        break
    else:
        small_df_ls.append(df[previous_step: step])
        previous_step = step
# unit testing
print ('no mistake on splitting based on shape' if pd.concat(small_df_ls, axis = 0).shape[0] == len(df)
       else 'oops, sth wrong in splitting')
print ('no mistakg on splltting based on length' if len(small_df_ls) == num_split 
       else "oops, sth wrong in splitting")

#---------------------------------------------
# Step2: imputatiing based on imputation method u chossed
#---------------------------------------------

def imputation(k):
	for drop_targtet in [True, False]: 
		#-------------------
		# imputating
		#------------------
		small_df_filled_ls = []
		for small_df in small_df_ls:
			# preprocessing for algorithm
			if drop_targtet == True:
				# avoidnig using TARGET to impute
				name = 'wo_target'
				X_missing = small_df.copy()
				X_missing.drop(['TARGET'], axis =1, inplace = True)
			else:
				name = 'w_target'
				X_missing = small_df.copy()
			# core algorithm: input should be array
			X_filled = KNN(k = k, verbose = True).complete(X_missing.values)
			small_df_filled_ls.append(pd.DataFrame(X_filled, columns = X_missing.columns))
		#-------------------
		# output
		#-------------------
		df_filled = pd.concat(small_df_filled_ls, axis = 0)
		#-------------------
		# save
		#-------------------
		output_path = '../features/filled'
		if not os.path.isdir(output_path):
			os.mkdir(output_path)

		df_filled.to_hdf(
			os.path.join(output_path, 'base_featurs_filled_knn_{}_{}.h5'.format(k, name)), 
			'base_featurs_filled_knn_{}_{}.h5'.format(k, name))

def multi(k):
	'''
	It's for using multi preprosessing to speed up the process.

	parameters:
	---------------------
	T: 5, 10, 15, 20
	'''
	imputation(k)

##################################################
# Main
##################################################
import multiprocessing as mp # for speeding up some process

s = time.time()

mp_pool = mp.Pool(4) # 4 == len of the below list
mp_pool.map(multi, [k for k in np.arange(5, 25, step = 5)]) 

e = time.time()
print (e-s, ' secs')
