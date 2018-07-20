import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('../input/application_train.csv')
    test_df = pd.read_csv('../input/application_test.csv')
    df = df.append(test_df).reset_index()
    df = df[['REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']]
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    print(df.sort_values(by=['REGION_POPULATION_RELATIVE']))