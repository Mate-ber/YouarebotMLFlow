import joblib
import numpy as np
import mlflow.pyfunc

from app.features import EMB_MODEL_NAME, featurize


class BotClassifier(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        from sentence_transformers import SentenceTransformer

        self.char_vec = joblib.load(context.artifacts["char_vectorizer"])
        self.word_vec = joblib.load(context.artifacts["word_vectorizer"])
        self.model = joblib.load(context.artifacts["model"])
        self.embedder = SentenceTransformer(EMB_MODEL_NAME)

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        X = featurize(texts, self.char_vec, self.word_vec, self.embedder)
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.005, 0.995)
