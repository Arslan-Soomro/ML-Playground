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


class LinearRegressionHyperParams(BaseModel):
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model")


class RegressionParams(BaseModel):
    config: ConfigParam | None = None
    hyper_params: dict | None = None


class LinearRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: LinearRegressionHyperParams


class AlgorithmResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metrics: RegMetrics
    predictions: Any
