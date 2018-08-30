import pandas as pd
import numpy as np

models = [
pd.read_csv("opensol-rank_mean-seed1115.csv"),
pd.read_csv("opensol-simple_mean.csv"),
pd.read_csv('sub-drop_null_hypo.csv'),
pd.read_csv('sub-drop_unused_in_half_folds.csv'),
pd.read_csv('sub-xgb_v0.csv'),
]

blend = models[0].copy()

for label in ["TARGET"]:
	ttlProb = 0
	for m in range(len(models)):
		ttlProb += models[m][label]
	blend[label] = ttlProb / len(models)

blend.to_csv('sub-simple_average.csv', index = False)


