from pydantic import BaseModel
#input text is string
class TextInput(BaseModel):
    text: str
#output is list of strings
class PredictionOut(BaseModel):
    predicted_genres: list[str]
