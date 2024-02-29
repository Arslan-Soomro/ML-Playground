from api.src.ml.utils.schemas import LinearRegressionParams, RidgeRegressionParams, SGDRegressorParams, ElasticNetHyperParams, LarsHyperParams, LassoHyperParams, LassoLarsHyperParams, ARDRegressionHyperParams

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
        "function": "ridge_regression",
    },
    "sgd_regression": {
        "id": 3,
        "title": "Stochastic Gradient Descent Regression",
        "description": "The SGDRegressor in scikit-learn is a linear model fitted by minimizing a regularized empirical loss with stochastic gradient descent, versatile for handling a variety of regression tasks.",
        "schema": SGDRegressorParams.model_json_schema(),
        "function": "sgd_regression",
    },
    "elastic_net_regression": {
        "id": 4,
        "title": "Elastic Net Regression",
        "description": "Elastic Net Regression combines L1 and L2 regularization to control complex models and prevent overfitting, useful when there are correlations between predictors.",
        "schema": ElasticNetHyperParams.model_json_schema(),
        "function": "elastic_net_regression",
    },
    "lars_regression": {
        "id": 5,
        "title": "Lars Regression",
        "description": "Lars Regression (Least Angle Regression) performs variable selection and regularization while being efficient for high-dimensional data.",
        "schema": LarsHyperParams.model_json_schema(),
        "function": "lars_regression",
    },
    "lasso_regression": {
        "id": 6,
        "title": "Lasso Regression",
        "description": "Lasso Regression applies L1 regularization, encouraging sparsity by driving some coefficients to zero, thus performing variable selection.",
        "schema": LassoHyperParams.model_json_schema(),
        "function": "lasso_regression",
    },
    "lasso_lars_regression": {
        "id": 7,
        "title": "LassoLars Regression",
        "description": "LassoLars Regression combines Lasso's sparsity with Lars's efficiency, ideal for high-dimensional datasets where variable selection is crucial.",
        "schema": LassoLarsHyperParams.model_json_schema(),
        "function": "lasso_lars_regression",
    },
    "ard_regression": {
        "id": 8,
        "title": "ARD Regression",
        "description": "ARD Regression (Automatic Relevance Determination) uses Bayesian regression to automatically determine relevant features, providing sparsity in the coefficients.",
        "schema": ARDRegressionHyperParams.model_json_schema(),
        "function": "ard_regression",
    },
}