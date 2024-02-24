from fastapi import APIRouter
from .ml_router import router as ml_router

router = APIRouter()

router.include_router(ml_router, prefix="/ml", tags=["ml"]);

@router.get("/")
async def root_api():
    return {"message": "Welcome to FastAPI API"}

@router.get("/items")
async def read_items():
    return [{"name": "Item One"}, {"name": "Item Two"}]

