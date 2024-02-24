import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Literal
from datetime import date
from api.src.ml.utils.dataset_utils import get_btc_data
from typing import TypeVar

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

    @field_validator('start_date')
    def date_must_be_before(cls, v, values):
        if v > values['end_date']:
            raise ValueError('start_date must be before end_date')
        if v < default_start_date:
            raise ValueError(f'start_date must be after {default_start_date}')

    @field_validator('end_date')
    def date_must_be_after(cls, v, values):
        if v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        if v > default_end_date:
            raise ValueError(f'end_date must be before {default_end_date}')


class RegMetrics(BaseModel):
    mse: float = Field(..., description="Mean Squared Error")
    r2: float = Field(..., description="R^2 Score")
    r2_percent: float = Field(..., description="R^2 Score in percentage")
    smape: float = Field(...,
                         description="Symmetric Mean Absolute Percentage Error")


class LinearRegressionHyperParams(BaseModel):
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model")


class LinearRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: LinearRegressionHyperParams


class AlgorithmResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metrics: RegMetrics
    predictions: pd.DataFrame
