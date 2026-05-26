"""Tools exposed to the LLM agent.

Each tool is a Python callable plus an OpenAI function-calling schema. The
agent picks tools at runtime based on the user's question.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.llm_rag.generator import RAGPipeline
from src.traditional_ml.predict import CreditRiskPredictor
from src.utils.logger import log


# ---------------------------- Tool implementations ---------------------------- #

def predict_credit_risk_tool(
    predictor: CreditRiskPredictor,
    loan_amnt: float,
    annual_inc: float,
    dti: float,
    int_rate: float = 12.0,
    grade: str = "C",
    purpose: str = "debt_consolidation",
    home_ownership: str = "RENT",
    emp_length: str = "5 years",
    term: str = "36 months",
    **kwargs: Any,
) -> dict:
    """Score an applicant."""
    applicant = {
        "loan_amnt": loan_amnt,
        "annual_inc": annual_inc,
        "dti": dti,
        "int_rate": int_rate,
        "grade": grade,
        "purpose": purpose,
        "home_ownership": home_ownership,
        "emp_length": emp_length,
        "term": term,
        **kwargs,
    }
    return predictor.predict(applicant)


def search_filings_tool(rag: RAGPipeline, query: str, top_k: int = 4) -> dict:
    """Search the RAG index over financial filings."""
    return rag.query(question=query, top_k=top_k)


def calculate_financial_ratios_tool(
    current_assets: float,
    current_liabilities: float,
    inventory: float = 0.0,
    total_debt: float = 0.0,
    total_equity: float = 1.0,
    net_income: float = 0.0,
    revenue: float = 1.0,
) -> dict:
    """Compute common financial ratios."""
    return {
        "current_ratio": round(current_assets / max(current_liabilities, 1e-6), 3),
        "quick_ratio": round(
            (current_assets - inventory) / max(current_liabilities, 1e-6), 3
        ),
        "debt_to_equity": round(total_debt / max(total_equity, 1e-6), 3),
        "net_profit_margin": round(net_income / max(revenue, 1e-6), 4),
        "interpretation": {
            "current_ratio": "> 1.5 is healthy; < 1.0 is concerning",
            "debt_to_equity": "> 2.0 indicates high leverage",
        },
    }


def generate_report_tool(
    company_name: str,
    summary: str,
    risk_score: float | None = None,
    recommendations: list[str] | None = None,
) -> dict:
    """Generate a structured analyst report."""
    return {
        "report": {
            "company": company_name,
            "executive_summary": summary,
            "risk_score": risk_score,
            "recommendations": recommendations or [],
            "disclaimer": "This is automated analysis, not investment advice.",
        }
    }


# ---------------------------- Schemas ---------------------------- #

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_credit_risk",
            "description": "Predict default probability for a loan applicant given their financial profile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amnt": {"type": "number", "description": "Requested loan amount in USD"},
                    "annual_inc": {"type": "number", "description": "Annual income in USD"},
                    "dti": {"type": "number", "description": "Debt-to-income ratio (%)"},
                    "int_rate": {"type": "number", "description": "Interest rate (%)"},
                    "grade": {"type": "string", "description": "Loan grade A–G"},
                    "purpose": {"type": "string", "description": "Loan purpose"},
                    "home_ownership": {"type": "string", "enum": ["RENT", "OWN", "MORTGAGE", "OTHER"]},
                    "emp_length": {"type": "string"},
                    "term": {"type": "string", "enum": ["36 months", "60 months"]},
                },
                "required": ["loan_amnt", "annual_inc", "dti"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Search SEC filings (10-K, 10-Q, annual reports) via RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_financial_ratios",
            "description": "Compute liquidity, leverage, profitability ratios from balance-sheet figures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_assets": {"type": "number"},
                    "current_liabilities": {"type": "number"},
                    "inventory": {"type": "number"},
                    "total_debt": {"type": "number"},
                    "total_equity": {"type": "number"},
                    "net_income": {"type": "number"},
                    "revenue": {"type": "number"},
                },
                "required": ["current_assets", "current_liabilities"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Produce a structured analyst report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "summary": {"type": "string"},
                    "risk_score": {"type": "number"},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["company_name", "summary"],
            },
        },
    },
]


def build_tool_registry(
    predictor: CreditRiskPredictor | None,
    rag: RAGPipeline | None,
) -> dict[str, Callable]:
    """Return name -> callable mapping bound to actual model/RAG instances."""
    registry: dict[str, Callable] = {
        "calculate_financial_ratios": calculate_financial_ratios_tool,
        "generate_report": generate_report_tool,
    }
    if predictor is not None:
        registry["predict_credit_risk"] = lambda **kw: predict_credit_risk_tool(predictor, **kw)
    if rag is not None:
        registry["search_filings"] = lambda **kw: search_filings_tool(rag, **kw)
    log.info(f"Registered tools: {list(registry.keys())}")
    return registry


def execute_tool(name: str, arguments: dict, registry: dict[str, Callable]) -> str:
    """Execute a tool by name with parsed JSON arguments. Returns a JSON string."""
    if name not in registry:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = registry[name](**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        log.exception(f"Tool {name} failed")
        return json.dumps({"error": str(e)})
