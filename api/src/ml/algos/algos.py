from api.src.ml.utils.schemas import LinearRegressionParams, RidgeRegressionParams, SGDRegressorParams

algos_collection = {
    "linear_regression": {
        "id": 1,
        "title": "Linear Regression",
        "description": "Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables.",
        "schema": LinearRegressionParams.model_json_schema(),
        "function": "linear_regression",
    },
    "ridge_regression": {
        "id": 2,
        "title": "Ridge Regression",
        "description": "Ridge Regression is a popular linear regression technique that aims to mitigate the issue of multicollinearity (high correlation between predictor variables) and overfitting in predictive modeling",
        "schema": RidgeRegressionParams.model_json_schema(),
        "function": "linear_regression",
    },
    "sgd_regression": {
        "id": 3,
        "title": "Stochastic Gradient Descent Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
    "elastic_net_regression": {
        "id": 4,
        "title": "Elastic Net Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
    "lars_regression": {
        "id": 5,
        "title": "Lars Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
    "lasso_regression": {
        "id": 6,
        "title": "Lasso Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
    "lasso_lars_regression": {
        "id": 7,
        "title": "LassoLars Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
    "ard_regression": {
        "id": 8,
        "title": "ARD Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "linear_regression",
    },
}