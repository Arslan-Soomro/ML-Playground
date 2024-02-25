from api.src.ml.algorithms.linear_regression_algo import linear_regression
from api.src.ml.utils.schemas import LinearRegressionParams, ConfigParam, LinearRegressionHyperParams


results = linear_regression(
    LinearRegressionParams(
        config=ConfigParam(test_size=0.2),
        hyper_params=LinearRegressionHyperParams()
    )
)

print(results["predictions"])