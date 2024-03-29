import pandas as pd

btc_dataset_path = "./api/src/ml/datasets/BTC.csv"

# Date Format: YYYY/MM/DD
def get_btc_data(startDate = None, endDate = None, remove_date_column = True):
    df = pd.read_csv(btc_dataset_path);
    
    df['date'] = pd.to_datetime(df['date'])
    
    if startDate and endDate:
       df = df[(df['date'] >= pd.to_datetime(startDate)) & (df['date'] <= pd.to_datetime(endDate))]
    
    df.reset_index(drop=True, inplace=True);
    
    retain_columns = df.columns.to_list();
    if(remove_date_column): retain_columns.remove("date");

    return df