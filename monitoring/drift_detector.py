import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import requests

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/creditcard.csv")
CURRENT_DATA_PATH = os.getenv("CURRENT_DATA_PATH", "data/live_logs/predictions.csv")
GITHUB_DISPATCH_URL = os.getenv("GITHUB_DISPATCH_URL", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def detect_drift():
    if not os.path.exists(REFERENCE_DATA_PATH):
        print(f"Reference data not found at {REFERENCE_DATA_PATH}")
        sys.exit(1)
        
    if not os.path.exists(CURRENT_DATA_PATH):
        print(f"Current data not found at {CURRENT_DATA_PATH}")
        sys.exit(1)

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    # Need at least a few samples to detect drift
    if len(current_data) < 10:
        print("Not enough recent data to calculate drift robustly. Exiting.")
        sys.exit(0)

    # We match common columns to avoid errors if log adds 'timestamp'
    common_columns = list(set(reference_data.columns) & set(current_data.columns))
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data[common_columns], current_data=current_data[common_columns])
    
    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    
    print(f"Dataset Drift Detected: {dataset_drift}")
    
    if dataset_drift:
        print("Data drift detected. Triggering retraining workflow.")
        trigger_retraining()
        # Exit with a special code to let GH Actions know drift occurred
        sys.exit(1)
    else:
        print("No significant data drift detected.")

def trigger_retraining():
    if not GITHUB_DISPATCH_URL or not GITHUB_TOKEN:
        print("GitHub Webhook URL or Token not configured. Could not trigger retrain using Webhook.")
        return
        
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    data = {
        "event_type": "trigger-retrain"
    }
    
    try:
        response = requests.post(GITHUB_DISPATCH_URL, json=data, headers=headers)
        if response.status_code == 204:
            print("Successfully triggered retrain workflow via repository dispatch.")
        else:
            print(f"Failed to trigger retrain: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error triggering webhook: {e}")

if __name__ == "__main__":
    detect_drift()
