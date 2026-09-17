"""
run_evaluation.py
-----------------
Loads processed validation and test datasets, scores them using the saved models,
and computes anomaly metrics. Updates docs/experiment_log.md.
"""
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.evaluation.metrics import anomaly_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_evaluation")

def main():
    logger.info("Loading processed datasets...")
    val_df = pd.read_csv("data/processed/val_processed.csv")
    test_df = pd.read_csv("data/processed/test_processed.csv")
    
    _NON_FEATURE_COLS = {
        "game_id", "draw_id", "machine_id", "ball_set_id",
        "venue_id", "draw_local_datetime", "season",
        "anomaly_score", "anomaly_flag", "risk_category",
    }
    
    def get_X(df):
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in _NON_FEATURE_COLS]
        return np.nan_to_num(df[feat_cols].values, nan=0.0)

    X_val = get_X(val_df)
    X_test = get_X(test_df)
    
    logger.info("Loading models...")
    iso_detector = joblib.load("models/iso_forest_detector.joblib")
    zscore_detector = joblib.load("models/zscore_detector.joblib")
    
    logger.info("Scoring validation set...")
    val_scores_iso = iso_detector.score(X_val)
    val_scores_zscore = zscore_detector.score(X_val)
    
    logger.info("Scoring test set...")
    test_scores_iso = iso_detector.score(X_test)
    test_scores_zscore = zscore_detector.score(X_test)
    
    threshold = 0.5
    z_threshold = 3.0
    
    val_summary = anomaly_summary(val_scores_iso, threshold)
    test_summary = anomaly_summary(test_scores_iso, threshold)
    val_z_summary = anomaly_summary(val_scores_zscore, z_threshold)
    
    logger.info(f"Isolation Forest Validation Summary: {val_summary}")
    logger.info(f"Isolation Forest Test Summary: {test_summary}")
    
    log_path = Path("docs/experiment_log.md")
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        new_row = f"| {today} | Isolation Forest | contam=0.05 | - | Anomaly Rate: {val_summary['flag_rate']:.2%} | Std: {val_summary['std_score']:.2f} |\n"
        if "| Date | Model | Config |" in content or "| Date" in content:
            content += new_row
            log_path.write_text(content, encoding="utf-8")
            logger.info("Updated docs/experiment_log.md")
        
    print("\n" + "="*40)
    print("📊 EVALUATION RESULTS SUMMARY")
    print("="*40)
    print(f"Isolation Forest [Validation]: {val_summary['flag_rate']:.2%} anomalies detected.")
    print(f"Isolation Forest [Test]      : {test_summary['flag_rate']:.2%} anomalies detected.")
    print(f"Robust Z-Score [Validation]  : {val_z_summary['flag_rate']:.2%} anomalies detected.")
    print("="*40)
    print("All metrics logged to docs/experiment_log.md\n")

if __name__ == "__main__":
    main()
