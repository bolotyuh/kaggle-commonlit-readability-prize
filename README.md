# API service for `commonlitreadabilityprize`
>
> Kaggle problem <https://www.kaggle.com/c/commonlitreadabilityprize/overview>

## Setup

### Define env variables

```bash
cp .env.example .env
```

```bash
CLEARML_API_ACCESS_KEY=
CLEARML_API_SECRET_KEY=
CLEARML_PROJECT=
CLEARML_TASK=
```

### Install requirements

```bash
conda create -n nlp_api python=3.10
conda activate nlp_api
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install --install-hooks
```

## Train model

Config file `conf/baseline.json`

Prepare dataset:

```bash
make prepare_dataset
```

Run train script:

```bash
make train
```

## Inference

### Usage

```bash
make serve
```

### API Docs

```bash
open http://localhost:8080/docs
```

### Demo app

```bash
open http://localhost:8080/demo
```

### Load test

```bash
locust -f locustfile.py --host=http://localhost:8080 --users 5 --spawn-rate 10 --autostart --run-time 300s
```
