from fastapi import APIRouter, HTTPException, Depends, Request
from api.src.ml.algorithms.algos import algos_collection
from api.src.ml.utils.schemas import LinearRegressionParams, AlgoOutput, ConfigParam, LinearRegressionHyperParams, AlgoRouteResponse
from api.src.ml.algorithms.linear_regression_algo import linear_regression

router = APIRouter()


@router.get("/algos", tags=["ml"], response_description="List of available algorithms")
def get_algos():
    return algos_collection


@router.get("/algos/{name}", tags=["ml"], response_description="Details of a specific algorithm")
def get_algo(name: str):
    if (name in algos_collection):
        return algos_collection[name]
    else:
        raise HTTPException(status_code=404, detail="Algorithm not found")


@router.post("/algos/compute/linear_regression", tags=["ml"], response_description="Compute the Linear Regression algorithm")
def compute_alog(payload: LinearRegressionParams) -> AlgoRouteResponse:
    try:
        metrics, predictions = linear_regression(
            payload
        )
        return {
            'data': {
                "metrics": metrics,
                "predictions": predictions.to_dict() # Pass orient="records" to return a list of dictionaries each representing a row
            },
            'message': "Success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
