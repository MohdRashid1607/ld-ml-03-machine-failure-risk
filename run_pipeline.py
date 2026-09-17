"""
run_pipeline.py
---------------
End-to-end orchestration script for the Machine Failure Risk project.

This script:
1. Loads and validates the raw CSV data.
2. Dynamically calculates chronological split dates and splits the data.
3. Builds the leakage-safe feature matrix.
4. Cleans, imputes, encodes, and scales the features.
5. Trains the Robust Z-Score Baseline and Isolation Forest models.
6. Saves the trained models and preprocessing artifacts.
"""

import logging
import os
import pandas as pd
from pathlib import Path

from src.data.loader import load_csv
from src.data.splitter import chronological_split
from src.data.transformer import clean_and_encode
from src.features.pipeline import build_feature_matrix
from src.models.trainer import train_and_save_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_pipeline")

def main():
    # 1. Load Data
    csv_path = "data/raw/lottery_weather_dataset_20260914 (3).csv"
    logger.info(f"Starting pipeline with data from {csv_path}")
    df = load_csv(csv_path)

    # 2. Chronological Split (Dynamically calculated based on actual dates)
    # 70% Train, 15% Val, 15% Test
    dt_col = df["draw_local_datetime"]
    train_end = dt_col.quantile(0.7).strftime('%Y-%m-%d %H:%M:%S')
    val_end = dt_col.quantile(0.85).strftime('%Y-%m-%d %H:%M:%S')
    
    logger.info(f"Calculated split boundaries - Train End: {train_end}, Val End: {val_end}")
    train_df, val_df, test_df = chronological_split(df, train_end=train_end, val_end=val_end)

    # 3. Feature Engineering (Strictly on training data to avoid leakage)
    logger.info("Building feature matrix for training data...")
    train_features = build_feature_matrix(train_df, fit=True)
    
    # 4. Cleaning, Encoding & Scaling
    logger.info("Cleaning, encoding, and scaling training features...")
    train_processed = clean_and_encode(train_features, fit=True)

    # Extract numeric matrix for unsupervised anomaly detection
    # Exclude non-numeric/passthrough columns from X_train
    _NON_FEATURE_COLS = {
        "game_id", "draw_id", "machine_id", "ball_set_id",
        "venue_id", "draw_local_datetime", "season",
    }
    numeric_cols = [
        c for c in train_processed.select_dtypes(include=['number']).columns 
        if c not in _NON_FEATURE_COLS
    ]
    X_train = train_processed[numeric_cols].values
    
    logger.info(f"Final training matrix shape: {X_train.shape}")

    # 5 & 6. Train and Save Models
    logger.info("Training Isolation Forest and Robust Z-Score models...")
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    results = train_and_save_models(
        X_train=X_train,
        output_dir=output_dir,
        iso_contamination=0.05,
        iso_n_estimators=100,
        iso_random_state=42
    )
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Models saved in: {output_dir}")
    
    # Save processed test/val sets for later evaluation
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process val and test sets (fit=False to use saved scalers/encoders)
    val_features = build_feature_matrix(val_df, fit=False)
    val_processed = clean_and_encode(val_features, fit=False)
    val_processed.to_csv(processed_dir / "val_processed.csv", index=False)
    
    test_features = build_feature_matrix(test_df, fit=False)
    test_processed = clean_and_encode(test_features, fit=False)
    test_processed.to_csv(processed_dir / "test_processed.csv", index=False)
    
    logger.info("Validation and Test processed datasets saved to data/processed/")

if __name__ == "__main__":
    main()
