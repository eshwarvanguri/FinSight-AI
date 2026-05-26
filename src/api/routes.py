"""FastAPI route handlers."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from src.agent.agent import FinSightAgent
from src.api.schemas import (
    AgentQuery,
    AgentResponse,
    ApplicantRequest,
    HealthResponse,
    PredictionResponse,
    RAGQuery,
    RAGResponse,
    RAGSource,
)
from src.llm_rag.generator import RAGPipeline
from src.traditional_ml.predict import CreditRiskPredictor
from src.utils.logger import log


router = APIRouter(prefix="/api/v1")

# Singleton holders (loaded on startup; see main.py)
_state: dict = {"predictor": None, "rag": None, "agent": None}


def get_predictor() -> CreditRiskPredictor:
    if _state["predictor"] is None:
        raise HTTPException(status_code=503, detail="Credit risk model not loaded")
    return _state["predictor"]


def get_rag() -> RAGPipeline:
    if _state["rag"] is None:
        raise HTTPException(status_code=503, detail="RAG index not loaded")
    return _state["rag"]


def get_agent() -> FinSightAgent:
    if _state["agent"] is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return _state["agent"]


# ---------------------------- Endpoints ---------------------------- #

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services={
            "credit_model": _state["predictor"] is not None,
            "rag": _state["rag"] is not None,
            "agent": _state["agent"] is not None,
        },
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    applicant: ApplicantRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict default probability for a loan applicant."""
    log.info(f"Predict request for loan_amnt={applicant.loan_amnt}")
    try:
        result = predictor.predict(applicant.model_dump())
    except Exception as e:
        log.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    return PredictionResponse(**result)


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    query: RAGQuery,
    rag: RAGPipeline = Depends(get_rag),
) -> RAGResponse:
    """Query the RAG index over financial filings."""
    log.info(f"RAG query: {query.question}")
    try:
        result = rag.query(question=query.question, top_k=query.top_k,
                           llm_model=query.llm_model)
    except Exception as e:
        log.error(f"RAG query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")
    return RAGResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[RAGSource(**s) for s in result["sources"]],
    )


@router.post("/agent/chat", response_model=AgentResponse)
async def agent_chat(
    query: AgentQuery,
    agent: FinSightAgent = Depends(get_agent),
) -> AgentResponse:
    """Run the full agent over a user query."""
    log.info(f"Agent query: {query.query}")
    try:
        result = agent.run(query.query)
    except Exception as e:
        log.error(f"Agent run failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    return AgentResponse(**result)
