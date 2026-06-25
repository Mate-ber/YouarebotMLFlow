import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000/predict")
LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")

app = FastAPI(title="orchestrator")


class IncomingMessage(BaseModel):
    text: str | None = None
    dialog_id: str
    id: str
    participant_index: int


class GetMessageRequest(BaseModel):
    dialog_id: str
    last_msg_text: str
    last_message_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(msg: IncomingMessage):
    try:
        resp = requests.post(CLASSIFIER_URL, json=msg.model_dump(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"classifier unavailable: {exc}")


@app.post("/get_message")
def get_message(body: GetMessageRequest):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful, brief chat assistant replying in Russian."},
            {"role": "user", "content": body.last_msg_text or "Привет!"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"llm unavailable: {exc}")
    return {"new_msg_text": reply, "dialog_id": body.dialog_id}