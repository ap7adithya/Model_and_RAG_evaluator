# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import setup_environment
from app.api.endpoints import model_evaluator, rag_evaluator

# Initialize environment
config = setup_environment()

# Create FastAPI app
app = FastAPI(
    title="GenAI Evaluator API",
    description="API endpoints for Model and RAG evaluation",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefixes
app.include_router(
    model_evaluator.router,
    prefix="/api/model-evaluator",
    tags=["Model Evaluator"]
)

app.include_router(
    rag_evaluator.router,
    prefix="/api/rag-evaluator",
    tags=["RAG Evaluator"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to GenAI Evaluator API"}