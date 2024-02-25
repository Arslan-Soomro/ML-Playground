from fastapi import APIRouter, HTTPException, Depends, Request
from api.src.ml.algorithms.algos import algos_collection
from api.src.ml.utils.schemas import LinearRegressionParams, AlgorithmResults, ConfigParam, LinearRegressionHyperParams
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

# FIXME config is None for some reason in both cases either when passed separately or as a single object


@router.post("/algos/compute/linear_regression", tags=["ml"], response_description="Compute the Linear Regression algorithm")
# -> AlgorithmResults:
def compute_alog(payload: LinearRegressionParams):
    # Catch any exceptions and return a 500 error
    # print("Params: ", config_r, hyper_params);
    try:
        print("Params: ");
        result = linear_regression(
            payload
        )
        return { result }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
