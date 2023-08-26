include .env

.EXPORT_ALL_VARIABLES:
CLEARML_LOG_MODEL = TRUE

check_dirs := scripts src serving

help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Run pre-commit hooks
	pre-commit run -a

prepare_dataset: ## Prepare dataset
	@python scripts/prepare_dataset.py ./data/raw/train.csv ./data \
		--test-size 0.15

.PHONY: train
train: ## Train model
	python -m src.train ./conf/baseline.json

.PHONY: serve
serve: ## Serve model (FastAPI)
	uvicorn --host 0.0.0.0 --port 8080 --workers 2 serving.api:app
