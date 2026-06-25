import json
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from app.features import EMB_MODEL_NAME, featurize

ARTIFACT_DIR = "app/artifacts"

print("Loading datasets...")
with open("data/train.json", "r", encoding="utf-8") as f:
    dialogs = json.load(f)
labels = pd.read_csv("data/ytrain.csv")

texts, y, groups = [], [], []
for _, row in labels.iterrows():
    dialog_id, participant = row["dialog_id"], row["participant_index"]
    if dialog_id not in dialogs:
        continue
    msgs = [m["text"] for m in dialogs[dialog_id]
            if m["participant_index"] == participant and m.get("text")]
    texts.append(" ".join(msgs).strip() or "привет")
    y.append(int(row["is_bot"]))
    groups.append(dialog_id)
y = np.array(y)
groups = np.array(groups)
print(f"Examples: {len(texts)}  (bot={int(y.sum())}, human={int((1 - y).sum())})")


char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2, max_features=8000, lowercase=True).fit(texts)
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           min_df=2, max_features=20000, lowercase=True).fit(texts)
print(f"Loading embedder '{EMB_MODEL_NAME}' and encoding {len(texts)} texts...")
embedder = SentenceTransformer(EMB_MODEL_NAME)
X = featurize(texts, char_vec, word_vec, embedder)


def make_model():
    return LogisticRegression(max_iter=2000, solver="liblinear",
                              class_weight="balanced", C=4.0, random_state=42)


print("\nRunning 5-fold grouped cross validation...")
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for train_idx, val_idx in skf.split(X, y, groups):
    model = make_model().fit(X[train_idx], y[train_idx])
    oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
oof = np.clip(oof, 0.005, 0.995)

print("\n=== CV METRICS (char+word TF-IDF + keyword feats + embeddings) ===")
print(f"ROC-AUC:  {roc_auc_score(y, oof):.4f}")
print(f"LogLoss:  {log_loss(y, oof):.4f}")
print(f"F1@0.5:   {f1_score(y, (oof > 0.5).astype(int)):.4f}")
print("\n" + classification_report(y, (oof > 0.5).astype(int), target_names=["Human", "Bot"]))

print("Training final model on all data...")
final_model = make_model().fit(X, y)

os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(char_vec, f"{ARTIFACT_DIR}/char_vectorizer.pkl")
joblib.dump(word_vec, f"{ARTIFACT_DIR}/word_vectorizer.pkl")
joblib.dump(final_model, f"{ARTIFACT_DIR}/model.pkl")
with open(f"{ARTIFACT_DIR}/meta.json", "w", encoding="utf-8") as f:
    json.dump({"emb_model_name": EMB_MODEL_NAME}, f)
print(f"Saved artifacts to {ARTIFACT_DIR}/ "
      f"(char_vectorizer, word_vectorizer, model, meta.json)")
