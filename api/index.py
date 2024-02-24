from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.src.routers import api_router

app = FastAPI();

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router.router, prefix="/api", tags=["api"]);

@app.get("/", tags=["root"])
async def root():
    return {"message": "You have reached the root route"}