import joblib

# load saved model files
model = joblib.load("model_store/genre_model.pkl")
tfidf = joblib.load("model_store/tfidf.pkl")
mlb = joblib.load("model_store/mlb.pkl")

def predict_genre(text: str):
    # transform text
    X = tfidf.transform([text])

    # make prediction
    pred = model.predict(X)

    # convert to labels
    labels = mlb.inverse_transform(pred)[0]
    
    return list(labels)

