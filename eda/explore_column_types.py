import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('../input/installments_payments.csv')
    # df = df.append(pd.read_csv('../input/application_test.csv')).reset_index()
    # df = df[['SK_ID_PREV','SK_ID_CURR','NUM_INSTALMENT_VERSION']]
    df = df[(df['SK_ID_PREV'] == 2234264) & (df['SK_ID_CURR'] == 184693)]
    # df = df.groupby('SK_ID_CURR').agg('sum')
    # df.drop_duplicates(subset=None, keep='first', inplace=True)
    df = df.sort_values(by=['DAYS_INSTALMENT'])
    print(df)