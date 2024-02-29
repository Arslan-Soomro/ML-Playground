import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from api.src.ml.utils.dataset_utils import get_btc_data
from api.src.ml.utils.utils import evaluate_reg_metrics, train_test_split_chronological, generate_predictions_df
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNet, Lars, Lasso, LassoLars, ARDRegression
from api.src.ml.utils.schemas import LinearRegressionParams, AlgoOutput, RidgeRegressionParams, SGDRegressorParams, ElasticNetParams, LarsParams, LassoParams, LassoLarsParams, ARDRegressionParams


# Linear Regression Algorithm
def linear_regression(args: LinearRegressionParams) -> AlgoOutput:    
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



# Ridge Regression Algorithm
def ridge_regression(args: RidgeRegressionParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create Ridge regression object
    model = Ridge(**hyper_params.dict())
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();


# SGD Regression Algorithm
def sgd_regression(args: SGDRegressorParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create SGD regression object
    model = SGDRegressor(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();

# Elastic Net Regression Algorithm
def elastic_net_regression(args: ElasticNetParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create Elastic Net regression object
    model = ElasticNet(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();

# Lars Regression Algorithm
def lars_regression(args: LarsParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create Lars regression object
    model = Lars(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();

# Lasso Regression Algorithm
def lasso_regression(args: LassoParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create Lasso regression object
    model = Lasso(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();

# LassoLars Regression Algorithm
def lasso_lars_regression(args: LassoLarsParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create LassoLars regression object
    model = LassoLars(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();

# ARD Regression Algorithm
def ard_regression(args: ARDRegressionParams) -> AlgoOutput:    
    config = args.config;
    hyper_params = args.hyper_params;
    
    # Date Format: YYYY/MM/DD
    df = get_btc_data(config.start_date, config.end_date)
    

    x = df[[*config.x_features]]
    y = df[config.y_feature]

    # Split the data into training/testing sets
    # Don't shuffle the data, we want to keep the chronological order, and we want to test on the most recent data
    x_train, x_test, y_train, y_test = train_test_split_chronological(x, y, config.test_size)
    
    # Create ARD regression object
    model = ARDRegression(**hyper_params.dict()) 
    model.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(x_test)
    

    metrics = evaluate_reg_metrics(y_test, y_pred)
    dates_of_predictions = get_btc_data(config.start_date, config.end_date, False)["date"].apply(lambda x: date(x.year, x.month, x.day));
    prediction_result = generate_predictions_df(x_test, y_test, y_pred, dates_of_predictions);

    return metrics, prediction_result.reset_index();