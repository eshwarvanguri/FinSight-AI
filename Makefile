.PHONY: help install setup data train index api test docker clean lint

help:
	@echo "FinSight AI — Common Commands"
	@echo ""
	@echo "  make install     Install Python dependencies"
	@echo "  make setup       Full setup (install + data + train + index)"
	@echo "  make data        Generate synthetic data + sample filings"
	@echo "  make train       Train the credit risk model"
	@echo "  make index       Build the RAG index"
	@echo "  make api         Run the FastAPI server (dev mode)"
	@echo "  make test        Run pytest"
	@echo "  make docker      Build and run with docker-compose"
	@echo "  make lint        Run ruff"
	@echo "  make clean       Remove generated artifacts"

install:
	pip install -r requirements.txt

setup: install data train index
	@echo "✓ Full setup complete. Run 'make api' to start."

data:
	python scripts/download_data.py --dataset lending_club --n 20000
	python scripts/download_data.py --dataset sample_10k

train:
	bash scripts/train_model.sh

index:
	bash scripts/build_index.sh

api:
	bash scripts/run_api.sh

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

docker:
	docker-compose -f docker/docker-compose.yml up --build

lint:
	ruff check src/ tests/

clean:
	rm -rf artifacts/models/*.pkl artifacts/embeddings/* artifacts/reports/*
	rm -rf data/raw/* data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
