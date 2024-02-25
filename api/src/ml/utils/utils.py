import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date

def symmetrical_mape(A, P):
    tmp = 2 * np.abs(P - A) / (np.abs(A) + np.abs(P))
    len_ = np.count_nonzero(~np.isnan(tmp))
    if len_ == 0 and np.nansum(tmp) == 0: # Deals with a special case
        return 100
    return 100 / len_ * np.nansum(tmp)

def evaluate_reg_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    smape = symmetrical_mape(y_true, y_pred)
    return {
        "mse": mse,
        "r2": r2,
        "r2_percent": r2 * 100,
        "smape": smape, # Symmetrical Mean Absolute Percentage Error
    }
    
def to_date_dict(dateStr):
    dateList = list(map(int, dateStr.split('/')))
    return {
        "year": dateList[0],
        "month": dateList[1],
        "day": dateList[2]
    }
    
def train_test_split_chronological(x, y, test_size):
    x_train = x.iloc[:-int(len(x) * test_size)]
    x_test = x.iloc[-int(len(x) * test_size):]
    y_train = y.iloc[:-int(len(y) * test_size)]
    y_test = y.iloc[-int(len(y) * test_size):]
    return x_train, x_test, y_train, y_test

def generate_predictions_df(x_test, y_test, y_pred, dates):
    predictions = x_test.copy();
    predictions['price'] = y_test;
    predictions['predicted_price'] = y_pred;
    # dates parameter is a df with only the date column, add it to the predictions df, horizontally
    predictions['date'] = dates;
    
    # Old couldn't use because of the use of year, month, day, which could not be present in the dataset
    # predictions['date'] = predictions[['year', 'month', 'day']].apply(lambda x: date(year=x['year'], month=x['month'], day=x['day']), axis=1);
    
    return predictions