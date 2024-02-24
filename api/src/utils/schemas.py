import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Literal
from datetime import date
from utils.dataset_utils import get_btc_data
from typing import TypeVar

btc_data = get_btc_data();

# Assume chronological order
default_start_date = date(year=btc_data.iloc[0]["year"], month=btc_data.iloc[0]["month"], day=btc_data.iloc[0]["day"]);
default_end_date = date(year=btc_data.iloc[-1]["year"], month=btc_data.iloc[-1]["month"], day=btc_data.iloc[-1]["day"]);


# It would remain mostly the same, specially for btc_data
class ConfigParam(BaseModel):
    start_date: date = default_start_date
    end_date: date = default_end_date
    x_features: List[Literal['open', 'high', 'low', 'vol',
                             'year', 'month', 'day']] = Field(default=['open', 'high', 'low', 'vol',
                             'year', 'month', 'day'], min_items=1, max_items=7)
    y_feature: Literal['price'] = 'price'
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)

    @field_validator('start_date')
    def date_must_be_before(cls, v, values):
        if v > values['end_date']:
            raise ValueError('start_date must be before end_date')

class RegMetrics(BaseModel):
    mse: float
    r2: float
    r2_percent: float
    smape: float

class LinearRegressionHyperParams(BaseModel):
    fit_intercept: bool = True


class LinearRegressionParams(BaseModel):
    config: ConfigParam
    hyper_params: LinearRegressionHyperParams
    

class AlgorithmResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    metrics: RegMetrics
    predictions: pd.DataFrame