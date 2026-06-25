import os
import time

import mlflow
from mlflow.tracking import MlflowClient

from app.features import EMB_MODEL_NAME
from app.mlflow_model import BotClassifier

MODEL_NAME = "bot-classifier"
ALIAS = "champion"
ARTIFACT_DIR = "app/artifacts"


def _register():
    mlflow.set_experiment("bot-classifier")
    with mlflow.start_run(run_name="champion-registration") as run:
        mlflow.log_params({"model_type": "LogisticRegression", "C": 4.0,
                           "embedding_model": EMB_MODEL_NAME})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BotClassifier(),
            artifacts={
                "char_vectorizer": f"{ARTIFACT_DIR}/char_vectorizer.pkl",
                "word_vectorizer": f"{ARTIFACT_DIR}/word_vectorizer.pkl",
                "model": f"{ARTIFACT_DIR}/model.pkl",
            },
            code_paths=["app"],
            registered_model_name=MODEL_NAME,
        )
    client = MlflowClient()
    version = client.search_model_versions(f"run_id='{run.info.run_id}'")[0].version
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)


def load_champion():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    for _ in range(30):
        try:
            client.search_experiments(max_results=1)
            break
        except Exception:
            time.sleep(3)

    try:
        client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        _register()

    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
