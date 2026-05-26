# FinSight AI

A financial risk and document intelligence platform. It combines a credit risk model trained on LendingClub data, a RAG pipeline over SEC 10-K filings, and an LLM agent that can call both as tools to answer analyst questions.

> Built as a learning project. Not investment advice. The credit model is trained on a public dataset and the RAG corpus is a small set of filings — useful for exploring the patterns, not for production lending decisions.

---

## What it does

- **Credit risk scoring** — predicts default probability for a loan applicant using an XGBoost + Random Forest ensemble with engineered features (debt-to-income, credit utilization, income-to-loan ratio, credit history length)
- **Document Q&A over 10-K filings** — natural-language questions answered by an LLM grounded in retrieved passages from SEC filings, with citations
- **Analyst agent** — an LLM agent that picks between the credit model, the RAG index, and a financial ratio calculator to answer multi-step questions like *"Assess credit risk for this applicant and contextualize it with their employer's latest filings"*
- **REST API** — everything exposed through FastAPI with OpenAPI docs

---

## Tech stack

**ML:** scikit-learn, XGBoost, SHAP, MLflow, SMOTE, isotonic calibration
**RAG:** sentence-transformers, FAISS, OpenAI / Anthropic APIs
**API:** FastAPI, Pydantic, Uvicorn
**Infra:** Docker, docker-compose

The credit model and the RAG pipeline are separate modules, each with its own training/build step. The agent layer sits on top and calls them through a tools interface.

---

## Architecture

```
                User / Analyst query
                        │
                        ▼
                  FastAPI layer
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   LLM Agent       RAG Pipeline   Credit Risk Model
  (orchestrator)   (10-K filings)  (XGBoost + RF)
        │               │               │
        │               ▼               ▼
        │         FAISS Vector     SHAP / Features
        │           Store
        │
        ▼
  Tools: predict_risk, search_filings,
         calculate_ratios, generate_report
```

---

## Project structure

```
finsight-ai/
├── data/
│   ├── raw/                  # LendingClub CSV, 10-K PDFs
│   ├── processed/            # Cleaned, feature-engineered
│   └── synthetic/
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
│   │   └── generator.py
│   ├── agent/                # LLM agent
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── prompts.py
│   ├── api/                  # FastAPI service
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes.py
│   └── utils/
├── configs/
│   ├── model_config.yaml
│   ├── rag_config.yaml
│   └── agent_config.yaml
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── artifacts/
│   ├── models/
│   ├── embeddings/
│   └── reports/
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.10+, ~8 GB RAM, and an OpenAI or Anthropic API key (or a local LLM via Ollama).

```bash
git clone https://github.com/eshwarvanguri/FinSight-AI.git
cd FinSight-AI
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MLFLOW_TRACKING_URI=./mlruns
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

Download sample data:

```bash
python scripts/download_data.py --dataset lending_club
python scripts/download_data.py --dataset sample_10k
```

---

## Running it

### Train the credit risk model

```bash
python -m src.traditional_ml.train \
    --config configs/model_config.yaml \
    --data data/raw/lending_club.csv \
    --output artifacts/models/credit_risk_v1.pkl \
    --use-mlflow
```

Common flags:

| Flag | Description | Default |
|---|---|---|
| `--config` | Path to YAML model config | `configs/model_config.yaml` |
| `--data` | Path to training CSV | required |
| `--output` | Where to save the model | `artifacts/models/model.pkl` |
| `--model-type` | `xgboost` / `random_forest` / `ensemble` | `ensemble` |
| `--use-mlflow` | Log to MLflow | `False` |
| `--tune` | Run Optuna hyperparameter tuning | `False` |
| `--n-trials` | Optuna trials | `50` |

### Evaluate

```bash
python -m src.traditional_ml.evaluate \
    --model artifacts/models/credit_risk_v1.pkl \
    --test-data data/processed/test.csv \
    --output artifacts/reports/eval_report.json \
    --generate-shap
```

### Build the RAG index

```bash
python -m src.llm_rag.ingestion \
    --input-dir data/raw/filings/ \
    --index-path artifacts/embeddings/finsight_index \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --embedding-model BAAI/bge-small-en-v1.5 \
    --vector-store faiss
```

### Query the RAG pipeline

```bash
python -m src.llm_rag.generator \
    --query "What were the main risk factors in the latest 10-K?" \
    --index-path artifacts/embeddings/finsight_index \
    --top-k 5 \
    --llm-model gpt-4o-mini
```

### Run the agent

```bash
python -m src.agent.agent \
    --query "Assess credit risk for applicant 12345 and add context from their employer's latest filings" \
    --config configs/agent_config.yaml \
    --verbose
```

### Launch the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

**Endpoints:**

- `POST /api/v1/predict` — credit risk prediction
- `POST /api/v1/rag/query` — query the document index
- `POST /api/v1/agent/chat` — full agentic conversation
- `GET /api/v1/health` — health check

### Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Spins up the API on `:8000`, MLflow on `:5000`, and Redis on `:6379`.

### Tests

```bash
pytest tests/ -v --cov=src
```

---

## Sample results

Credit model performance on the LendingClub test split:

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Logistic Regression | 0.712 | 0.34 | 0.184 |
| Random Forest | 0.748 | 0.41 | 0.171 |
| XGBoost (tuned) | 0.782 | 0.46 | 0.158 |
| Stacked Ensemble | 0.779 | 0.45 | 0.159 |

---

## Datasets

- **LendingClub Loan Data** (Kaggle) — credit risk labels and applicant features
- **SEC EDGAR filings** — 10-K, 10-Q reports, downloaded via `sec-edgar-downloader`
- **Synthetic applicants** — generated for stress-testing the model

---

## Possible improvements

- Add streaming responses to the agent
- Cache repeated RAG queries (Redis)
- Add a small web frontend instead of just the API
- Periodic re-training pipeline
- Data drift monitoring (PSI, KS test) in production

---

## License

MIT
