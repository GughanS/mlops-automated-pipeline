from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import os
import datetime
from api.schemas import TransactionFeature, PredictionResponse
from api.dependencies import load_model, _model

app = FastAPI(title="Credit Card Fraud Prediction API", version="1.0.0")

# Mount frontend
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Setup logging path
LOG_DIR = os.getenv("LOG_DIR", "data/live_logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "predictions.csv")

@app.on_event("startup")
def startup_event():
    """Instantiate MLflow client and load the production model."""
    load_model()
    if LOG_FILE_PATH and not os.path.exists(LOG_FILE_PATH):
        # Initialize CSV with headers
        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "is_fraud", "timestamp"]
        df = pd.DataFrame(columns=cols)
        df.to_csv(LOG_FILE_PATH, index=False)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: TransactionFeature):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or unavailable")
    
    # Create a DataFrame for the prediction
    data = features.model_dump()
    df = pd.DataFrame([data])
    
    # Drop Time as the model was trained without it
    if 'Time' in df.columns:
        df_for_pred = df.drop(columns=['Time'])
    else:
        df_for_pred = df
        
    try:
        prediction = model.predict(df_for_pred)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Log to live_logs
    data["is_fraud"] = int(prediction)
    data["timestamp"] = datetime.datetime.now().isoformat()
    log_df = pd.DataFrame([data])
    log_df.to_csv(LOG_FILE_PATH, mode="a", header=False, index=False)
    
    return PredictionResponse(is_fraud=int(prediction))
