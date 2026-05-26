"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApplicantRequest(BaseModel):
    loan_amnt: float = Field(..., gt=0, description="Loan amount in USD")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (%)")
    int_rate: float = Field(12.0, ge=0)
    grade: str = Field("C")
    sub_grade: str | None = None
    home_ownership: str = Field("RENT")
    purpose: str = Field("debt_consolidation")
    emp_length: str = Field("5 years")
    term: str = Field("36 months")
    revol_util: float | None = Field(None, ge=0)
    delinq_2yrs: int = Field(0, ge=0)
    open_acc: int = Field(5, ge=0)
    total_acc: int = Field(15, ge=0)
    mort_acc: int = Field(0, ge=0)
    pub_rec: int = Field(0, ge=0)
    pub_rec_bankruptcies: int = Field(0, ge=0)
    inq_last_6mths: int = Field(0, ge=0)
    addr_state: str = Field("CA")
    verification_status: str = Field("Verified")
    installment: float | None = None


class PredictionResponse(BaseModel):
    probability_of_default: float
    verdict: str
    risk_tier: str
    threshold_used: float


class RAGQuery(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)
    use_reranker: bool = True
    llm_model: str = "llama-3.3-70b-versatile"


class RAGSource(BaseModel):
    source: str | None
    page: int | str | None
    score: float | None
    snippet: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]


class AgentQuery(BaseModel):
    query: str = Field(..., min_length=3)
    max_iterations: int = Field(8, ge=1, le=15)
    verbose: bool = False


class AgentResponse(BaseModel):
    answer: str
    iterations: int
    trace: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, bool]
