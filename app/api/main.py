import os
from contextlib import asynccontextmanager
from uuid import uuid4

import joblib
import requests
from fastapi import FastAPI

from app.core.logging import app_logger
from app.features import EMB_MODEL_NAME, featurize
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction


CHAR_VECTORIZER_PATH = "app/artifacts/char_vectorizer.pkl"
WORD_VECTORIZER_PATH = "app/artifacts/word_vectorizer.pkl"
MODEL_PATH = "app/artifacts/model.pkl"
LLM_URL = os.getenv("LLM_URL", "http://llm:8080/v1/chat/completions")

char_vectorizer = None
word_vectorizer = None
embedder = None
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global char_vectorizer, word_vectorizer, embedder, model
    if all(os.path.exists(p) for p in (CHAR_VECTORIZER_PATH, WORD_VECTORIZER_PATH, MODEL_PATH)):
        from sentence_transformers import SentenceTransformer
        char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
        word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        embedder = SentenceTransformer(EMB_MODEL_NAME)
        app_logger.info("Classifier (TF-IDF + keyword feats + embeddings) loaded successfully.")
    else:
        app_logger.warning("Artifacts missing! Run 'python evaluate.py' to generate them.")
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness/readiness probe used by the youare.bot contract."""
    loaded = all(x is not None for x in (model, char_vectorizer, word_vectorizer, embedder))
    return {"status": "ok", "model_loaded": loaded}


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    This functions receives a message from HumanOrNot and returns a response
    """
    app_logger.info(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )

    user_prompt = body.last_msg_text if body.last_msg_text else "Привет!"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful, brief chat assistant replying in Russian language."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post(LLM_URL, json=payload, timeout=15)
        response.raise_for_status()
        llm_data = response.json()
        reply_text = llm_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        app_logger.error(f"Failed to communicate with LLM container: {str(e)}")
        reply_text = body.last_msg_text

    return GetMessageResponseModel(
        new_msg_text=reply_text, dialog_id=body.dialog_id
    )


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Endpoint to save a message and get the probability
    that this message if from bot .

    Returns a `Prediction` object.
    """
    text_to_analyze = msg.text if (msg.text and msg.text.strip()) else "привет"

    try:
        if model is None or char_vectorizer is None or word_vectorizer is None or embedder is None:
            is_bot_probability = 0.5
        else:
            X = featurize([text_to_analyze], char_vectorizer, word_vectorizer, embedder)
            raw_prob = float(model.predict_proba(X)[0, 1])
            is_bot_probability = min(max(raw_prob, 0.005), 0.995)
    except Exception as e:
        # Never fail a prediction — a 500 counts as an incorrect response on the platform.
        app_logger.error(f"Prediction failed, falling back to 0.5: {e}")
        is_bot_probability = 0.5

    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )