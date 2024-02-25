from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from api.src.ml.utils.schemas import LinearRegressionParams, AlgoOutput
from api.src.ml.utils.utils import evaluate_reg_metrics, train_test_split_chronological, generate_predictions_df
from api.src.ml.utils.dataset_utils import get_btc_data


def linear_regression(args: LinearRegressionParams) -> AlgoOutput:    
    # print("Args: ", args);
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]
    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create linear regression object
    model = LinearRegression(**hyper_params.dict())
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();