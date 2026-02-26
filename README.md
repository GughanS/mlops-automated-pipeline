# End-to-End MLOps Pipeline: Financial Fraud Detection

This repository contains an enterprise-grade Machine Learning Operations (MLOps) pipeline built to handle continuous deployment, monitoring, and automated retraining of a **Credit Card Fraud Detection** model on the real [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Usecase & Concept Drift
Financial behavior changes rapidly due to macroeconomic events. This pipeline is built to simulate a scenario similar to the COVID-19 pandemic. 
- **Baseline:** The Random Forest Classifier is initially trained on normal pre-pandemic consumer spending represented by the raw Kaggle dataframe. It utilizes `class_weight='balanced'` to detect the 0.17% minority fraud cases effectively.
- **Live Simulation:** Live inference requests streaming from a custom UI send 28-dimensional PCA Vectors (`V1-V28`) alongside `Time` and `Amount` fields.
- **Drift Handling:** Evidently AI actively monitors the resulting live inference data. If significant covariance or dataset drift is detected against the real baseline, it automatically triggers a GitHub Actions CI/CD retraining job to learn the "new normal" and promote the updated model via MLflow.

## Features
- **API & Serving:** FastAPI backend that serves predictions dynamically and scales inference arrays seamlessly.
- **Modern UI:** Glassmorphism-styled frontend supporting Light/Dark modes, asynchronous API calls, and one-click test injections.
- **Monitoring Layer:** Automatically detects data drift in live inference data natively using Evidently AI.
- **Model Training:** Automated Random Forest training script with MLflow tracking and Model Registry stage promotion.
- **Orchestration:** GitHub Actions pipelines that run scheduled drift checks natively.

## Getting Started

### Local Development
1. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```
2. The UI and FastAPI App will be available at `http://localhost:8000`.
3. The MLflow tracking server will be available at `http://localhost:5000`.

### Submitting Predictions Manually (cURL)
Send a POST request to `/predict` targeting the Fraud endpoint with a full 30-feature dict:
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"Time": 0, "Amount": 149.62, "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781, "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637, "V10": 0.0907, "V11": -0.5515, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210}'
```
