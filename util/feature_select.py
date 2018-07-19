import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def display_distributions(df, col1, col2):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)

    # Plot col1 importances
    ax = plt.subplot(gs[0, 0])
    ax.hist(df[col1].values, bins=100, range=(df[col1].min(),min(df[col1].max(),50)))
    ax.set_title('Split %s Distribution'%col1)
    plt.xlabel('importance')

    # Plot col2 importances
    ax = plt.subplot(gs[0, 1])
    ax.hist(df[col2].values, bins='auto', range=(df[col2].min(),min(df[col2].max(),100)))
    ax.set_title('Split %s Distribution'%col2)
    plt.xlabel('importance')
    plt.show()

feat_imp_df = pd.read_csv('/Users/Allen/MEGA/HCDR/logs/ver8/feature_importance-drop.csv')
null_feat_imp_df = pd.read_csv('/Users/Allen/MEGA/HCDR/logs/ver8/feature_importance-null_hypo-drop-level_wise_tree.csv')

null_feat_imp_df.columns = ['feature', 'null_importance']
feat_imp_df = pd.merge(feat_imp_df, null_feat_imp_df, on=['feature'])

# feat_imp_df['importance_clipped'] = feat_imp_df['importance'].clip(upper = 100)
# display_distributions(feat_imp_df, 'importance', 'null_importance')

feat_imp_df['null_importance'] = feat_imp_df['null_importance'].replace({0 : 0.0001})

feat_imp_df['importance_to_null_importance'] = feat_imp_df['importance'] / feat_imp_df['null_importance']
feat_imp_df = feat_imp_df.sort_values(by=['importance_to_null_importance'])
bad_feature_list = list(feat_imp_df.loc[(feat_imp_df['importance_to_null_importance'] <= 13) & (feat_imp_df['importance_to_null_importance'] > 12)]['feature'])

print("Features to remove:", bad_feature_list)