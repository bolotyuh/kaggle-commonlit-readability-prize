import logging
import os
import shutil
from pathlib import Path
from typing import List

import torch
from clearml import Task
from filelock import FileLock
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logger = logging.getLogger(__name__)


MODEL_CACHE_DIR = Path.home().joinpath(".cache", "fastapi_models")


def load_from_clearml(project_name: str, task_name: str, model_path: Path) -> Path:
    task = Task.get_task(project_name=project_name, task_name=task_name, tags=["best", "production"])
    best_snapshot = None

    for m in task.models["output"]:
        if "production" in m.tags:
            best_snapshot = m
            break

    assert best_snapshot is not None, "Not found model"

    if not model_path.exists():
        logger.info(f"Copy model to `{model_path}`")
        shutil.copytree(best_snapshot.get_local_copy(), model_path)

    return model_path


class Predictor:
    def __init__(self, model_load_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_load_path)
        self.model.eval()

        logger.info(f"Loaded model from `{model_load_path}`")

    @torch.no_grad()
    def predict(self, text: List[str]):
        text_encoded = self.tokenizer.batch_encode_plus(list(text), return_tensors="pt", padding=True)

        return self.model(**text_encoded).logits.flatten().numpy()

    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        project_name = os.getenv("CLEARML_PROJECT", None)
        task_name = os.getenv("CLEARML_TASK", None)

        assert project_name or task_name, 'You have to define "CLEARML_PROJECT" and "CLEARML_TASK" ENV variable'

        MODEL_PATH = MODEL_CACHE_DIR.joinpath(project_name, task_name)
        with FileLock(".model.lock"):
            if not MODEL_PATH.joinpath("pytorch_model.bin").exists():
                load_from_clearml(project_name=project_name, task_name=task_name, model_path=MODEL_PATH)

        return cls(model_load_path=MODEL_PATH)
