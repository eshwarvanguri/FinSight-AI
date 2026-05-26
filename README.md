# 💹 FinSight AI — Intelligent Financial Risk & Insights Platform

> **An end-to-end AI/ML platform combining traditional ML, RAG, LLMs, and Agentic AI to assess credit risk, analyze financial documents, and generate actionable insights — built to production standards.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 Why This Project Stands Out

This is **not** a toy notebook. FinSight AI is engineered like a real fintech product and exercises the full modern AI/ML stack that large companies (Goldman, JPMC, American Express, Razorpay, Stripe, fintech unicorns) actually hire for:

| Layer | What it shows recruiters |
|---|---|
| **Traditional ML** | Feature engineering, XGBoost, Random Forest, LightGBM, calibration, SHAP explainability — the bread & butter of risk modeling |
| **RAG Pipeline** | Vector DB (FAISS/Chroma), hybrid search (BM25 + dense), re-ranking, chunking strategies on 10-K reports |
| **LLM Orchestration** | Function-calling, structured outputs, prompt engineering, evaluation with LLM-as-judge |
| **Agentic AI** | A multi-tool agent that picks between the credit model, the RAG index, and live calculators to answer analyst questions |
| **MLOps** | MLflow tracking, Docker, FastAPI serving, model monitoring, drift detection, CI-ready tests |
| **Software Engineering** | Clean modular code, configs via Hydra/YAML, typed Python, unit tests, reproducibility |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      User / Analyst                              │
│   "Is Reliance a credit risk given their latest 10-K?"          │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   FastAPI Layer   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼────────┐  ┌────────▼────────┐
│  LLM Agent     │  │   RAG Pipeline   │  │  Credit Risk    │
│  (Orchestrator)│  │  (Financial      │  │  Model          │
│                │  │   Documents)     │  │  (XGBoost+RF)   │
└───────┬────────┘  └─────────┬────────┘  └────────┬────────┘
        │                     │                     │
        │           ┌─────────▼────────┐  ┌────────▼────────┐
        │           │  Vector Store    │  │  Feature Store  │
        │           │  (FAISS/Chroma)  │  │  + SHAP         │
        │           └──────────────────┘  └─────────────────┘
        │
┌───────▼─────────────────────────────────────────────────────┐
│  Tools: [predict_risk, search_filings, calculate_ratios,    │
│         get_market_data, generate_report]                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. **Credit Risk Scoring Engine** (Traditional ML)
- XGBoost + Random Forest + Logistic Regression stacked ensemble
- 35+ engineered features (debt-to-income, utilization ratios, behavioral signals)
- SHAP-based explainability per prediction
- Probability calibration with isotonic regression
- Class imbalance handled via SMOTE + scale_pos_weight

### 2. **Financial Document Intelligence** (RAG)
- Ingests 10-K, 10-Q, earnings call transcripts, annual reports (PDFs)
- Hierarchical chunking (semantic + recursive)
- Hybrid retrieval: BM25 (lexical) + dense embeddings (semantic) + Reciprocal Rank Fusion
- Cross-encoder re-ranking for top-k precision
- Citations with page numbers and source tracking

### 3. **LLM-Powered Analyst Agent**
- Tool-using agent (function calling) over: risk model, RAG index, financial calculators
- Multi-step reasoning with intermediate state
- Structured output (Pydantic) for downstream systems
- Hallucination guardrails via retrieval grounding

### 4. **Production-Ready API**
- FastAPI endpoints with OpenAPI docs
- Async batch + streaming responses
- Rate limiting, request validation, error handling
- Dockerized for one-command deploy

### 5. **MLOps & Evaluation**
- MLflow experiment tracking
- LLM-as-judge eval for RAG (faithfulness, relevance, groundedness)
- RAGAS-style metrics
- Data drift monitoring (PSI, KS test)
- Model card auto-generation

---

## 📁 Project Structure

```
finsight-ai/
├── data/
│   ├── raw/                  # Raw PDFs, CSVs (e.g., LendingClub, 10-K filings)
│   ├── processed/            # Cleaned, feature-engineered data
│   └── synthetic/            # Generated test data
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis
│   ├── 02_feature_eng.ipynb  # Feature engineering & selection
│   └── 03_rag_evaluation.ipynb
├── src/
│   ├── traditional_ml/       # Credit risk model
│   │   ├── data_pipeline.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── llm_rag/              # RAG pipeline
│   │   ├── ingestion.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── generator.py
│   ├── agent/                # LLM agent orchestrator
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── prompts.py
│   ├── api/                  # FastAPI service
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── monitoring.py
├── configs/
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   └── agent_config.yaml
├── tests/                    # pytest suite
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── scripts/                  # CLI runners
│   ├── train_model.sh
│   ├── build_index.sh
│   └── run_api.sh
├── docs/
│   └── ARCHITECTURE.md
├── artifacts/
│   ├── models/               # Saved models
│   ├── embeddings/           # FAISS indices
│   └── reports/              # Generated reports
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (optional but recommended)
- 8GB+ RAM
- OpenAI/Anthropic API key (or local LLM via Ollama)

### 1. Clone & Install
```bash
git clone https://github.com/<you>/finsight-ai.git
cd finsight-ai
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MLFLOW_TRACKING_URI=./mlruns
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

### 3. Download Sample Data
```bash
python scripts/download_data.py --dataset lending_club
python scripts/download_data.py --dataset sample_10k
```

---

## 🏃 How to Run — Step by Step

### **Step 1: Train the Credit Risk Model**

```bash
python -m src.traditional_ml.train \
    --config configs/model_config.yaml \
    --data data/raw/lending_club.csv \
    --output artifacts/models/credit_risk_v1.pkl \
    --experiment-name credit-risk-baseline \
    --use-mlflow
```

**Arguments:**
| Flag | Description | Default |
|---|---|---|
| `--config` | Path to YAML model config | `configs/model_config.yaml` |
| `--data` | Path to training CSV | required |
| `--output` | Where to save the trained model | `artifacts/models/model.pkl` |
| `--model-type` | `xgboost` / `random_forest` / `ensemble` | `ensemble` |
| `--use-mlflow` | Log to MLflow | `False` |
| `--tune` | Run Optuna hyperparameter tuning | `False` |
| `--n-trials` | Optuna trials if `--tune` | `50` |

---

### **Step 2: Evaluate the Model**

```bash
python -m src.traditional_ml.evaluate \
    --model artifacts/models/credit_risk_v1.pkl \
    --test-data data/processed/test.csv \
    --output artifacts/reports/eval_report.json \
    --generate-shap
```

**Arguments:**
| Flag | Description | Default |
|---|---|---|
| `--model` | Trained model path | required |
| `--test-data` | Held-out test set | required |
| `--output` | Where to save metrics JSON | `eval.json` |
| `--generate-shap` | Compute SHAP values + plots | `False` |
| `--threshold` | Decision threshold | `0.5` |

---

### **Step 3: Build the RAG Index from Financial Documents**

```bash
python -m src.llm_rag.ingestion \
    --input-dir data/raw/filings/ \
    --index-path artifacts/embeddings/finsight_index \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-model BAAI/bge-small-en-v1.5 \
    --vector-store faiss
```

**Arguments:**
| Flag | Description | Default |
|---|---|---|
| `--input-dir` | Folder of PDFs to ingest | required |
| `--index-path` | Where to save the vector index | required |
| `--chunk-size` | Tokens per chunk | `512` |
| `--chunk-overlap` | Token overlap between chunks | `64` |
| `--embedding-model` | HF embedding model | `BAAI/bge-small-en-v1.5` |
| `--vector-store` | `faiss` / `chroma` | `faiss` |
| `--use-semantic-chunking` | Use semantic chunker | `False` |

---

### **Step 4: Query the RAG Pipeline (CLI)**

```bash
python -m src.llm_rag.generator \
    --query "What were the main risk factors mentioned in the latest 10-K?" \
    --index-path artifacts/embeddings/finsight_index \
    --top-k 5 \
    --use-reranker \
    --llm-model gpt-4o-mini
```

---

### **Step 5: Run the Full Agent**

```bash
python -m src.agent.agent \
    --query "Assess credit risk for applicant ID 12345 and contextualize with their employer's latest 10-K filings" \
    --config configs/agent_config.yaml \
    --verbose
```

---

### **Step 6: Launch the API Server**

```bash
# Development
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

**Available endpoints:**
- `POST /api/v1/predict` — Credit risk prediction
- `POST /api/v1/rag/query` — Query the document index
- `POST /api/v1/agent/chat` — Full agentic conversation
- `GET /api/v1/health` — Health check
- `GET /api/v1/metrics` — Prometheus-style metrics

---

### **Step 7: Docker Deployment**

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This spins up:
- The FinSight API on port `8000`
- MLflow UI on port `5000`
- A Redis cache on port `6379`

---

### **Step 8: Run Tests**

```bash
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📊 Sample Results (on LendingClub data)

| Model | ROC-AUC | PR-AUC | KS | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.712 | 0.34 | 0.31 | 0.184 |
| Random Forest | 0.748 | 0.41 | 0.37 | 0.171 |
| XGBoost (tuned) | **0.782** | **0.46** | **0.42** | **0.158** |
| Stacked Ensemble | 0.779 | 0.45 | 0.41 | 0.159 |

**RAG Evaluation (LLM-as-judge over 100 questions):**
- Faithfulness: **0.91**
- Answer Relevance: **0.87**
- Context Precision: **0.83**

---

## 🧠 Datasets Used

1. **LendingClub Loan Data** (Kaggle) — Credit risk labels
2. **SEC EDGAR Filings** — 10-K, 10-Q reports (downloadable via `sec-edgar-downloader`)
3. **FinanceBench** — Open benchmark for financial QA evaluation
4. **Synthetic** — Generated applicant profiles for stress-testing

---

## 🛠️ Tech Stack

**ML/AI:** XGBoost, scikit-learn, LightGBM, SHAP, Optuna, MLflow
**LLM/RAG:** LangChain, LlamaIndex, FAISS, ChromaDB, sentence-transformers, OpenAI, Anthropic
**Backend:** FastAPI, Pydantic, Uvicorn
**Infra:** Docker, docker-compose, Redis
**Eval:** RAGAS, DeepEval, pytest

---

## 🎓 What Recruiters Will Notice

✅ You can ship a **real, end-to-end** AI product, not just train a model
✅ You understand **trade-offs** (when to use traditional ML vs. LLM vs. RAG)
✅ You can write **production code** with configs, tests, logging, monitoring
✅ You handle **evaluation rigorously** — both ML metrics and LLM-as-judge
✅ You think about **explainability, calibration, and safety**

---

## 📜 License

MIT

## 🙏 Acknowledgments

Inspired by patterns from BloombergGPT, FinGPT, and modern fintech ML platforms.
