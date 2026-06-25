import numpy as np
from scipy.sparse import csr_matrix, hstack

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BOT_MARKERS = [
    "as an ai", "ai assistant", "language model", "chatgpt", "openai", "gpt",
    "virtual assistant", "how can i assist", "i can help", "i cannot", "i can't",
    "i do not have", "i don't have", "i am a bot",
    "я бот", "я чат-бот", "чат-бот", "виртуальный помощник", "искусственный интеллект",
    "как могу помочь", "чем могу помочь", "не могу", "не имею", "мои возможности", "обращаться",
]

HUMAN_STYLE_MARKERS = [
    "лол", "ахах", "хаха", "бро", "чел", "ну", "ок", "спс", "не бот", "я человек",
]


def handcrafted(text: str) -> list[float]:
    n = len(text)
    lower = text.lower()
    toks = text.split()
    ntok = max(len(toks), 1)
    cyr = sum(("а" <= c.lower() <= "я") or c.lower() == "ё" for c in text)
    lat = sum("a" <= c.lower() <= "z" for c in text)
    return [
        n,
        ntok,
        n / ntok,
        sum(c.isupper() for c in text) / max(n, 1),
        text.count("?"),
        text.count("!"),
        sum(c.isdigit() for c in text) / max(n, 1),
        len(set(toks)) / ntok,
        cyr / max(n, 1),
        lat / max(n, 1),
        sum(m in lower for m in BOT_MARKERS),
        sum(m in lower for m in HUMAN_STYLE_MARKERS),
    ]


def featurize(texts, char_vec, word_vec, embedder) -> csr_matrix:
    Xc = char_vec.transform(texts)
    Xw = word_vec.transform(texts)
    Xh = csr_matrix(np.asarray([handcrafted(t) for t in texts], dtype=np.float64))
    Xe = csr_matrix(embedder.encode(list(texts), normalize_embeddings=True))
    return hstack([Xc, Xw, Xh, Xe]).tocsr()
