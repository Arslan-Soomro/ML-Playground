from api.src.ml.utils.schemas import LinearRegressionParams

algos_collection = {
    "linear_regression": {
        "id": 1,
        "title": "Linear Regression",
        "description": "Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables.",
        "schema": LinearRegressionParams.model_json_schema(),
        "function": "linear_regression",
    },
}