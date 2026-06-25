from contextlib import asynccontextmanager
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI

from app.mlflow_bootstrap import load_champion
from app.models import IncomingMessage, Prediction

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_champion()
    yield


app = FastAPI(title="classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    text = msg.text if (msg.text and msg.text.strip()) else "привет"
    prob = float(state["model"].predict(pd.DataFrame({"text": [text]}))[0])
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=prob,
    )
