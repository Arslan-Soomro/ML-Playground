import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import List, Literal
from datetime import date
from api.src.ml.utils.dataset_utils import get_btc_data
from typing import TypeVar, Any

btc_data = get_btc_data()

# Assume chronological order
default_start_date = date(
    year=btc_data.iloc[0]["year"], month=btc_data.iloc[0]["month"], day=btc_data.iloc[0]["day"])
default_end_date = date(
    year=btc_data.iloc[-1]["year"], month=btc_data.iloc[-1]["month"], day=btc_data.iloc[-1]["day"])


# It would remain mostly the same, specially for btc_data
class ConfigParam(BaseModel):
    start_date: date = Field(default=default_start_date,
                             description="Start date of the dataset")
    end_date: date = Field(default=default_end_date,
                           description="End date of the dataset")
    x_features: List[Literal['open', 'high', 'low', 'vol',
                             'year', 'month', 'day']] = Field(default=['open', 'high', 'low', 'vol',
                                                                       'year', 'month', 'day'], min_items=1, max_items=7, description="List of features to use as input")
    y_feature: Literal['price'] = Field('price', description="Output feature")
    test_size: float = Field(default=0.2, ge=0.0, le=1.0,
                             description="Size of the test set")

    @model_validator(mode="after")
    @classmethod
    def check_dates(cls, data):
        st_date = data.start_date
        ed_date = data.end_date
        if st_date > ed_date:
            raise ValueError('start_date must be before end_date')
        if st_date < default_start_date or st_date > default_end_date:
            raise ValueError(
                f'start_date must be after {default_start_date} and before {default_end_date}')
        if ed_date < default_start_date or ed_date > default_end_date:
            raise ValueError(
                f'end_date must be after {default_start_date} and before {default_end_date}')
        return data;


class RegMetrics(BaseModel):
    mse: float = Field(..., description="Mean Squared Error")
    r2: float = Field(..., description="R^2 Score")
    r2_percent: float = Field(..., description="R^2 Score in percentage")
    smape: float = Field(...,
                         description="Symmetric Mean Absolute Percentage Error")

class RegressionParams(BaseModel):
    config: ConfigParam | None = None
    hyper_params: dict | None = None

# Linear Regression Model Configuation
# Hyperparameters & Config
class LinearRegressionHyperParams(BaseModel):
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model")


class LinearRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: LinearRegressionHyperParams

# Ridge Regression Model Configuation
# Hyperparameters & Config
class RidgeRegressionHyperParams(BaseModel):
    alpha: float = Field(
        default=1.0, description="Constant that multiplies the L2 term, controlling regularization strength.")
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model.")
    tol: float = Field(
        default=1e-4, description="Precision of the solution.")
    solver: str = Field(
        default='auto', description="Solver to use in the computational routines. Options include 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'. 'auto' chooses the solver based on the type of data.")
    random_state: int = Field(
        default=None, description="Used when the solver is 'sag' or 'saga' to shuffle the data.", nullable=True)


class RidgeRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: RidgeRegressionHyperParams


# Stochastic Gradient Descent Model Configuation
# Hyperparameters & Config
class SGDRegressorHyperParams(BaseModel):
    loss: str = Field(
        default='squared_error', description="The loss function to be used. Defaults to 'squared_error'. Other options include 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.")
    penalty: str = Field(
        default='l2', description="The penalty (aka regularization term) to be used. Defaults to 'l2'. Other options include 'l1', 'elasticnet', and 'none'.")
    alpha: float = Field(
        default=0.0001, description="Constant that multiplies the regularization term. Defaults to 0.0001.")
    l1_ratio: float = Field(
        default=0.15, description="The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. Defaults to 0.15.")
    max_iter: int = Field(
        default=1000, description="The maximum number of passes over the training data (aka epochs). Defaults to 1000.")
    tol: float = Field(
        default=1e-3, description="The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol). Defaults to 1e-3.")
    learning_rate: str = Field(
        default='invscaling', description="The learning rate schedule. Options include 'constant', 'optimal', 'invscaling', and 'adaptive'. Defaults to 'invscaling'.")
    eta0: float = Field(
        default=0.01, description="The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules. Defaults to 0.01.")

class SGDRegressorParams(BaseModel):
    config: ConfigParam
    hyper_params: SGDRegressorHyperParams


# Elastic Net Model Configuation
# Hyperparameters & Config
class ElasticNetHyperParams(BaseModel):
    alpha: float = Field(
        default=1.0, description="Constant that multiplies the penalty terms. Defaults to 1.0.")
    l1_ratio: float = Field(
        default=0.5, description="The Elastic Net mixing parameter, with 0 < l1_ratio <= 1. Defaults to 0.5.")
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model. Defaults to True.")
    normalize: bool = Field(
        default=False, description="If True, the regressors X will be normalized before regression. Defaults to False.")
    max_iter: int = Field(
        default=1000, description="The maximum number of iterations. Defaults to 1000.")
    tol: float = Field(
        default=1e-4, description="The tolerance for the optimization. Defaults to 1e-4.")

class ElasticNetParams(BaseModel):
    config: ConfigParam
    hyper_params: ElasticNetHyperParams


# Least Angle Regression Model Configuation
# Hyperparameters & Config
class LarsHyperParams(BaseModel):
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model. Defaults to True.")
    normalize: bool = Field(
        default=True, description="If True, the regressors X will be normalized before regression. Defaults to True.")
    precompute: bool = Field(
        default=False, description="Whether to use a precomputed Gram matrix to speed up calculations. Defaults to False.")
    n_nonzero_coefs: int = Field(
        default=500, description="The maximum number of non-zero coefficients. Defaults to 500.")
    eps: float = Field(
        default=2.220446049250313e-16, description="The machine precision for floating-point operations. Defaults to approximately 2.22e-16.")

class LarsParams(BaseModel):
    config: ConfigParam
    hyper_params: LarsHyperParams

# Lasso Regression Model Configuation
# Hyperparameters & Config
class LassoHyperParams(BaseModel):
    alpha: float = Field(
        default=1.0, description="Constant that multiplies the L1 term. Defaults to 1.0.")
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model. Defaults to True.")
    normalize: bool = Field(
        default=False, description="If True, the regressors X will be normalized before regression. Defaults to False.")
    max_iter: int = Field(
        default=1000, description="The maximum number of iterations. Defaults to 1000.")
    tol: float = Field(
        default=0.0001, description="The tolerance for the optimization. Defaults to 1e-4.")

class LassoParams(BaseModel):
    config: ConfigParam
    hyper_params: LassoHyperParams

# LassoLars Regression Model Configuation
# Hyperparameters & Config
class LassoLarsHyperParams(BaseModel):
    alpha: float = Field(
        default=1.0, description="Constant that multiplies the L1 term. Defaults to 1.0 for LassoLars.")
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model. Defaults to True for LassoLars.")
    normalize: bool = Field(
        default=True, description="If True, the regressors X will be normalized before regression. Defaults to True for LassoLars.")
    max_iter: int = Field(
        default=500, description="The maximum number of iterations. Adjusted to 500 for LassoLars.")
    eps: float = Field(
        default=2.220446049250313e-16, description="The machine precision for floating-point operations. Defaults to approximately 2.22e-16 for LassoLars.")

class LassoLarsParams(BaseModel):
    config: ConfigParam
    hyper_params: LassoLarsHyperParams

# ARD Regression Model Configuation
# Hyperparameters & Config
class ARDRegressionHyperParams(BaseModel):
    n_iter: int = Field(
        default=300, description="Maximum number of iterations. Defaults to 300.")
    tol: float = Field(
        default=1e-3, description="Stopping criterion. Defaults to 1e-3.")
    alpha_1: float = Field(
        default=1e-6, description="Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.")
    alpha_2: float = Field(
        default=1e-6, description="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter. Defaults to 1e-6.")
    lambda_1: float = Field(
        default=1e-6, description="Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.")
    lambda_2: float = Field(
        default=1e-6, description="Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter. Defaults to 1e-6.")
    compute_score: bool = Field(
        default=False, description="If True, compute the objective function at each step of the model. Defaults to False.")

class ARDRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: ARDRegressionHyperParams

class AlgoOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # metrics: RegMetrics
    # predictions: Any
    output: tuple[RegMetrics, pd.DataFrame]

class AlgoRouteResponseData(BaseModel):
    metrics: RegMetrics
    predictions: dict

class AlgoRouteResponse(BaseModel):
    data: AlgoRouteResponseData
    message: str
