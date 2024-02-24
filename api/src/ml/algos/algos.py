from utils.schemas import LinearRegressionParams, AlgorithmResults

algos = [
    {
        "id": 1,
        "title": "Linear Regression",
        "description": "Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables.",
        "schema": LinearRegressionParams.json(indent=2),
        "function": "linear_regression",
    }
]