import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from training.data_loader import load_data
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///" + os.path.abspath("data/mlruns").replace("\\", "/"))
MODEL_NAME = "FraudDetectionModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    mlflow.create_experiment("Default")
except:
    pass
mlflow.set_experiment("Default")

def train_and_evaluate():
    X, y = load_data()
    # If the combined dataset is too small, skip training
    if len(X) < 5:
        print("Not enough data to train. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if there is already a production model to compare against
    client = MlflowClient()
    prod_f1 = 0.0
    try:
        # Get production model metrics
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if latest_versions:
            prod_run_id = latest_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_f1 = prod_run.data.metrics.get("f1_score", 0.0)
            print(f"Current Production F1 Score: {prod_f1}")
    except Exception as e:
        print(f"No existing production model found or error fetching it: {e}")

    with mlflow.start_run() as run:
        # Hyperparameters
        n_estimators = int(os.getenv("N_ESTIMATORS", 100))
        max_depth = int(os.getenv("MAX_DEPTH", 10))
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Add class_weight='balanced' to handle the massive 0.17% fraud imbalance in the Kaggle set
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(clf, "model", registered_model_name=MODEL_NAME)
        
        print(f"New Model F1 Score: {f1}")
        
        if f1 > prod_f1:
            print("New model performs better. Promoting to Production.")
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, MODEL_NAME)
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Production",
                archive_existing_versions=True
            )
        else:
            print("New model did not improve upon Production model. Keeping current model.")

if __name__ == "__main__":
    train_and_evaluate()
