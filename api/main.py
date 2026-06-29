from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import os

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="AI-Driven Citizen Grievance & Sentiment Analysis System",
    description="Predicts government department and complaint priority from citizen grievance text.",
    version="1.0"
)

# -----------------------------
# Load Department Classification Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

department_model = joblib.load(os.path.join(MODEL_DIR, "department_model.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# -----------------------------
# Request Schema
# -----------------------------
class ComplaintRequest(BaseModel):
    complaint: str

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # remove special chars
    text = re.sub(r"\s+", " ", text).strip()              # remove extra spaces
    return text

# -----------------------------
# Rule-Based Priority Function
# -----------------------------
def predict_priority_from_text(text):
    text = str(text).lower()

    # Critical priority
    critical_keywords = [
        "no electricity", "power outage", "no power",
        "no heat", "no water", "gas leak", "fire", "smoke",
        "flood", "flooding", "sewer backup", "urgent", "emergency"
    ]

    # High priority
    high_keywords = [
        "water leakage", "water leak", "leakage", "ceiling leak",
        "street light", "traffic signal", "pothole",
        "electric issue", "electric problem", "rodent"
    ]

    # Medium priority
    medium_keywords = [
        "garbage", "missed collection", "noise",
        "illegal parking", "dirty", "unsanitary", "sanitation"
    ]

    if any(word in text for word in critical_keywords):
        return "Critical"
    elif any(word in text for word in high_keywords):
        return "High"
    elif any(word in text for word in medium_keywords):
        return "Medium"
    else:
        return "Low"

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Citizen Grievance AI API is running!",
        "usage": "Send a POST request to /predict with complaint text."
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_complaint(data: ComplaintRequest):
    raw_text = data.complaint
    cleaned = clean_text(raw_text)

    # Department prediction using ML model
    vectorized_text = tfidf_vectorizer.transform([cleaned])
    predicted_department = department_model.predict(vectorized_text)[0]

    # Priority prediction using rule-based engine
    predicted_priority = predict_priority_from_text(raw_text)

    return {
        "complaint": raw_text,
        "predicted_department": predicted_department,
        "predicted_priority": predicted_priority
    }