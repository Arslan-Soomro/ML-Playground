from fastapi import FastAPI, Request
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
# Define a middleware, catch all requests, log request body and pass control to next router
"""
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print("Request: ", await request.json());
    response = await call_next(request)
    return response
"""

app.include_router(api_router.router, prefix="/api", tags=["api"]);

@app.get("/", tags=["root"])
async def root():
    return {"message": "You have reached the root route"}