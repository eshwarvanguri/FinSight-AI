"""Tool-using LLM agent.

Implements a simple ReAct-style loop with OpenAI function calling. The agent
keeps calling tools until the LLM emits a final assistant message without a
tool call, or `max_iterations` is reached.

Usage:
    python -m src.agent.agent \\
        --query "Assess credit risk for applicant X and pull recent 10-K context" \\
        --config configs/agent_config.yaml \\
        --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agent.tools import TOOL_SCHEMAS, build_tool_registry, execute_tool
from src.llm_rag.generator import RAGPipeline
from src.traditional_ml.predict import CreditRiskPredictor
from src.utils.config import load_config, settings
from src.utils.logger import log


SYSTEM_PROMPT = """You are FinSight, a senior financial risk analyst AI.

You have access to these tools:
1. predict_credit_risk — score a borrower's default probability
2. search_filings — RAG search over SEC filings (10-K, 10-Q)
3. calculate_financial_ratios — compute liquidity/leverage ratios
4. generate_report — produce a structured analyst report

Rules:
- Always ground claims in evidence; cite filings when relevant.
- Use predict_credit_risk for any default-probability claim.
- Break down complex questions and use multiple tools as needed.
- Refuse to give investment advice; provide analysis only.
- End with a structured report when the question warrants it.
"""


class FinSightAgent:
    """Tool-using agent."""

    def __init__(
        self,
        model_path: str | None = None,
        index_path: str | None = None,
        llm_model: str | None = None,
        max_iterations: int = 8,
        temperature: float = 0.1,
        verbose: bool = False,
    ):
        self.llm_model = llm_model or settings.LLM_MODEL or "grok-3-mini"
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose

        predictor = CreditRiskPredictor.load(model_path) if model_path else None
        rag = RAGPipeline(index_path=index_path) if index_path else None
        self.tools = build_tool_registry(predictor, rag)

        from openai import OpenAI

        if settings.XAI_API_KEY:
            self.client = OpenAI(api_key=settings.XAI_API_KEY, base_url=settings.XAI_BASE_URL)
            log.info(f"Agent using xAI/Grok — model: {self.llm_model}")
        elif settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            log.info(f"Agent using OpenAI — model: {self.llm_model}")
        else:
            log.warning("No XAI_API_KEY or OPENAI_API_KEY set — agent will use stub mode.")
            self.client = None

    def run(self, query: str) -> dict[str, Any]:
        """Run the agent loop until completion or max_iterations."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        trace: list[dict] = []

        if self.client is None:
            return {
                "answer": "[Stub mode — set OPENAI_API_KEY]",
                "trace": trace,
                "iterations": 0,
            }

        for step in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=self.temperature,
            )
            msg = response.choices[0].message

            # Final answer (no tool calls)
            if not msg.tool_calls:
                if self.verbose:
                    log.info(f"Final answer at step {step + 1}")
                return {
                    "answer": msg.content,
                    "trace": trace,
                    "iterations": step + 1,
                }

            # Otherwise dispatch tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if self.verbose:
                    log.info(f"→ tool: {name}({args})")
                result = execute_tool(name, args, self.tools)
                trace.append({"tool": name, "arguments": args, "result_preview": result[:300]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return {
            "answer": "[max iterations reached without final answer]",
            "trace": trace,
            "iterations": self.max_iterations,
        }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the FinSight agent")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/agent_config.yaml")
    p.add_argument("--model-path", type=str, default="artifacts/models/credit_risk_v1.pkl")
    p.add_argument("--index-path", type=str, default="artifacts/embeddings/finsight_index")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    llm_model = cfg.get("agent", {}).get("llm_model", "gpt-4o")

    agent = FinSightAgent(
        model_path=args.model_path if Path(args.model_path).exists() else None,
        index_path=args.index_path if Path(args.index_path).exists() else None,
        llm_model=llm_model,
        verbose=args.verbose,
    )
    result = agent.run(args.query)
    print("\n" + "=" * 80)
    print("ANSWER:")
    print(result["answer"])
    print("\nIterations:", result["iterations"])
    if args.verbose:
        print("\nTRACE:")
        for step in result["trace"]:
            print(json.dumps(step, indent=2))


if __name__ == "__main__":
    main()
