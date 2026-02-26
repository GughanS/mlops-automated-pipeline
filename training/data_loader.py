import os
import pandas as pd

def load_data():
    """
    Loads baseline data and appends any live logs to create a full training dataset.
    """
    reference_path = os.getenv("REFERENCE_DATA_PATH", "data/creditcard.csv")
    live_logs_path = os.getenv("CURRENT_DATA_PATH", "data/live_logs/predictions.csv")
    
    if os.path.exists(reference_path):
        df_ref = pd.read_csv(reference_path)
    else:
        df_ref = pd.DataFrame()
        
    if os.path.exists(live_logs_path):
        df_live = pd.read_csv(live_logs_path)
    else:
        df_live = pd.DataFrame()
        
    # Drop timestamps before concat to avoid NaNs on reference data
    if 'timestamp' in df_ref.columns:
        df_ref = df_ref.drop(columns=['timestamp'])
    if 'timestamp' in df_live.columns:
        df_live = df_live.drop(columns=['timestamp'])
        
    # Combine datasets
    to_concat = []
    if not df_ref.empty:
        to_concat.append(df_ref)
    if not df_live.empty:
        to_concat.append(df_live)
        
    if not to_concat:
        raise ValueError("No data found to train on.")
        
    df_full = pd.concat(to_concat, ignore_index=True)
    
    # Drop rows with missing values
    df_full = df_full.dropna()
    
    # Clean the Kaggle target structure
    if 'Class' in df_full.columns:
        df_full = df_full.rename(columns={'Class': 'is_fraud'})

    # Separate features and target
    if 'is_fraud' not in df_full.columns:
        raise ValueError("Missing target column 'is_fraud' in the dataset.")
        
    # We drop timestamp explicitly, as the real kaggle set includes Time which isn't useful for predictive state
    if 'timestamp' in df_full.columns:
        df_full = df_full.drop(columns=['timestamp'])
    if 'Time' in df_full.columns:
        df_full = df_full.drop(columns=['Time'])
        
    X = df_full.drop(columns=['is_fraud'])
    y = df_full['is_fraud']
    
    return X, y
