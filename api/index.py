from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI();

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "You have reached the root route"}

@app.get("/api")
async def root_api():
    return {"message": "Welcome to FastAPI API"}

@app.get("/api/items")
async def read_items():
    return [{"name": "Item One"}, {"name": "Item Two"}]
