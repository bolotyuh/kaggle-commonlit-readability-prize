from typing import List

import gradio as gr
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from serving.gradio_ui import gradio_block
from serving.predictor import Predictor


load_dotenv(find_dotenv())


class Payload(BaseModel):
    text: List[str]


class Prediction(BaseModel):
    pred: List[float]


app = FastAPI()
predictor = Predictor.default_from_model_registry()
app = gr.mount_gradio_app(app, gradio_block(predictor), path="/demo")


@app.get("/health")
def health() -> str:
    return "success"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = predictor.predict(text=payload.text)

    return Prediction(pred=prediction.tolist())
