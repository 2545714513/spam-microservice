
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

app = FastAPI(title="SMS Spam Classifier API")

class MessageInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "SMS Spam Classifier API is running"}

@app.post("/predict")
def predict(data: MessageInput):
    transformed_text = vectorizer.transform([data.text])
    prediction = model.predict(transformed_text)[0]
    return {
        "input_text": data.text,
        "prediction": prediction
    }
}
