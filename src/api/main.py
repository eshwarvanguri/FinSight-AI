"""FastAPI application entrypoint.

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agent.agent import FinSightAgent
from src.api.routes import _state, router
from src.llm_rag.generator import RAGPipeline
from src.traditional_ml.predict import CreditRiskPredictor
from src.utils.config import settings
from src.utils.logger import log


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/credit_risk_v1.pkl")
INDEX_PATH = os.getenv("INDEX_PATH", "artifacts/embeddings/finsight_index")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup, release on shutdown."""
    log.info("FinSight API starting...")

    if Path(MODEL_PATH).exists():
        try:
            _state["predictor"] = CreditRiskPredictor.load(MODEL_PATH)
            log.info("✓ Credit risk model loaded")
        except Exception as e:
            log.error(f"Failed to load credit model: {e}")
    else:
        log.warning(f"Model not found at {MODEL_PATH} — /predict will be unavailable")

    if Path(INDEX_PATH).exists():
        try:
            _state["rag"] = RAGPipeline(index_path=INDEX_PATH)
            log.info("✓ RAG index loaded")
        except Exception as e:
            log.error(f"Failed to load RAG: {e}")
    else:
        log.warning(f"Index not found at {INDEX_PATH} — /rag will be unavailable")

    try:
        _state["agent"] = FinSightAgent(
            model_path=MODEL_PATH if Path(MODEL_PATH).exists() else None,
            index_path=INDEX_PATH if Path(INDEX_PATH).exists() else None,
        )
        log.info("✓ Agent initialized")
    except Exception as e:
        log.error(f"Failed to init agent: {e}")

    yield
    log.info("FinSight API shutting down.")


app = FastAPI(
    title="FinSight AI",
    description="Intelligent Financial Risk & Insights Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")
