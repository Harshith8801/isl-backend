from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import os
from collections import deque, Counter

app = FastAPI(title="ISL Local API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MEANINGS = {
    "GM": "Good Morning",
    "hello": "Hello",
    "how": "How are you?",
    "IMF": "I am Fine",
    "indian_india": "Indian / India",
    "language": "Language",
    "man": "Man",
    "none": "None",
    "sign": "Sign",
    "women": "Woman",
    "welcome": "Welcome",
}

DYNAMIC_FRAMES = 30
DYNAMIC_FEATURES = 126

prediction_history = deque(maxlen=3)

MODEL_PATH = "./model/lstm_model.final"
LABEL_PATH = "./model/dynamic_labels.json"

model = None
labels = []

try:
    from tensorflow.keras.models import load_model

    print("Loading model...")

    model = load_model(MODEL_PATH)

    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)

    labels = [label_map[str(i)] for i in range(len(label_map))]

    print("Model loaded successfully")
    print(labels)

except Exception as e:
    print("MODEL LOAD ERROR:")
    print(e)


class PredictRequest(BaseModel):
    sequence: list


@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "labels": labels
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(req: PredictRequest):

    if model is None:
        return {
            "prediction": "none",
            "meaning": "Model not loaded",
            "confidence": 0
        }

    try:
        seq = np.array(req.sequence, dtype=np.float32)

        if seq.ndim == 1:
            seq = seq.reshape(
                DYNAMIC_FRAMES,
                DYNAMIC_FEATURES
            )

        if seq.shape != (30, 126):
            return {
                "error": f"Expected shape (30,126), got {seq.shape}"
            }

        inp = seq.reshape(
            1,
            DYNAMIC_FRAMES,
            DYNAMIC_FEATURES
        )

        probs = model.predict(inp, verbose=0)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        prediction = labels[idx]

        prediction_history.append(prediction)

        final_prediction = Counter(
            prediction_history
        ).most_common(1)[0][0]

        return {
            "prediction": final_prediction,
            "meaning": LABEL_MEANINGS.get(
                final_prediction,
                final_prediction
            ),
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return {
            "error": str(e)
        }