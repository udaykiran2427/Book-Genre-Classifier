from fastapi import FastAPI
from app.predictor import predict_genre
from app.schema import TextInput, PredictionOut
from fastapi.middleware.cors import CORSMiddleware
# create the FastAPI app instance
app = FastAPI(title="Book Genre Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# endpoint: POST /predict
@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextInput):
    # payload.text is the input text from the JSON body
    genres = predict_genre(payload.text)
    return PredictionOut(predicted_genres=genres)

# simple health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
