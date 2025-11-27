import joblib
import numpy as np

# load saved model files
model = joblib.load("model_store/genre_model.pkl")
tfidf = joblib.load("model_store/tfidf.pkl")
mlb = joblib.load("model_store/mlb.pkl")

FALLBACK_MESSAGE = ["Could not determine genre. Provide more detailed text."]

def predict_genre(text: str):
    # 1. Very short input fallback
    if len(text.strip()) < 10:
        return FALLBACK_MESSAGE

    # transform text
    X = tfidf.transform([text])

    # 2. TF-IDF produced an empty vector (no known words)
    if X.nnz == 0:   # nnz = number of non-zero elements
        return FALLBACK_MESSAGE

    # predict
    pred = model.predict(X)

    # 3. No labels predicted (all zeros)
    if pred.sum() == 0:
        return FALLBACK_MESSAGE

    # convert predicted vector to labels
    labels = mlb.inverse_transform(pred)[0]

    # 4. Another final sanity check
    if not labels:
        return FALLBACK_MESSAGE

    return list(labels)
