from fastapi import APIRouter
from api.src.ml.algorithms.algos import algos_list

router = APIRouter()

@router.get("/algos", tags=["ml"], response_description="List of available algorithms")
def get_algos():
    return algos_list;

@router.get("/algos/{id}", tags=["ml"], response_description="Details of a specific algorithm")
def get_algo(id: int):
    return next((algo for algo in algos_list if algo["id"] == id), None);