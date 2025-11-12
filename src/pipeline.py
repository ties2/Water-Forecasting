# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#
# import torch
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# import yaml
# from src.data.ingest import DataIngestionService
# from src.data.preprocess import DataPreprocessor
# from src.features.build_features import FeatureEngineer
# from src.models.train import ModelBuilder
# from src.models.predict import DataTransformer
# from src.utils.logging import setup_logger
# import joblib
#
# logger = setup_logger("pipeline")
#
# def run_pipeline():
#     with open("config/config.yaml") as f:
#         config = yaml.safe_load(f)
#
#     # 1. Ingest
#     ingest = DataIngestionService(config)
#     flow = ingest.load_sensor_data("data/raw/flow_data.csv")
#     weather = ingest.fetch_weather_data(flow.index.min(), flow.index.max())
#     merged = ingest.merge_all_data(flow, weather)
#
#     # 2. Preprocess
#     pp = DataPreprocessor()
#     merged = pp.handle_missing_values(merged)
#
#     # 3. Features
#     fe = FeatureEngineer()
#     df = fe.engineer_all(merged, 'flow_rate_m3h')
#
#     # 4. Transform
#     dt = DataTransformer()
#     target = df['flow_rate_m3h'].values
#     features = df.drop(columns=['flow_rate_m3h']).values
#     X_scaled = dt.fit_scaler(features)
#     y_scaled = dt.fit_scaler(target)
#
#     X, y = dt.create_sequences(X_scaled, seq_len=168)
#
#     # 5. Train
#     model = ModelBuilder.build_lstm(168, X.shape[2])
#     # model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
#     # Example fix in pipeline.py
#     split_index = int(len(X) * 0.8)
#
#     X_train, y_train = X[:split_index], y[:split_index]
#     X_val, y_val = X[split_index:], y[split_index:]
#
#     # Pass validation data explicitly
#     model.fit(X_train, y_train,
#               epochs=50,
#               batch_size=32,
#               validation_data=(X_val, y_val),
#               shuffle=False)  # Optionally disable shuffling for safety, though it's on by default
#
#     # model.save("models/lstm_model.pkl")
#     model.save("models/lstm_model.h5")
#
#
#
#
#     joblib.dump(dt.scaler, "models/scaler.pkl")
#     logger.info("Pipeline completed!")

#new version

import os
import yaml
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# --- Custom Module Imports ---
# Assuming your project structure is src/
from src.data.ingest import DataIngestionService
from src.data.preprocess import DataPreprocessor, DataValidator
from src.features.build_features import FeatureEngineer
from src.models.train import ModelBuilder
from src.utils.logging import setup_logger

# --- Environment Setup ---
# Set thread counts to avoid resource contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# Import torch and set threads *after* os.environ
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --- Setup ---
logger = setup_logger("pipeline")
CONFIG_PATH = "config/config.yaml"
# This should match the sequence length used in your app.py
SEQ_LEN = 168


def create_sequences(features, target, seq_len=SEQ_LEN):
    """
    Creates sequences of data for time series forecasting.
    """
    xs, ys = [], []
    for i in range(len(features) - seq_len):
        x = features[i:(i + seq_len)]
        y = target[i + seq_len]
        xs.append(x)
        ys.append(y)

    logger.info(f"Created {len(xs)} sequences.")
    return np.array(xs), np.array(ys)


def run_pipeline():
    """
    Executes the full training pipeline.
    """
    logger.info("Starting pipeline run...")

    # --- 0. Load Configuration ---
    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {CONFIG_PATH}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {CONFIG_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    # --- 1. Ingest Data ---
    ingest = DataIngestionService(config)
    flow = ingest.load_sensor_data(config['data']['flow_data_path'])
    weather = ingest.fetch_weather_data(flow.index.min(), flow.index.max())
    merged = ingest.merge_all_data(flow, weather)

    # --- 2. Validate & Preprocess ---
    validator = DataValidator(config.get('validation_thresholds', {}))
    issues = validator.check_missing_values(merged)
    if issues['has_issues']:
        logger.warning("High percentage of missing data found. Proceeding with interpolation.")

    pp = DataPreprocessor()
    merged = pp.handle_missing_values(merged)
    logger.info(f"Data preprocessed. Shape: {merged.shape}")

    # --- 3. Feature Engineering ---
    fe = FeatureEngineer()
    target_col = 'flow_rate_m3h'
    df_features = fe.engineer_all(merged, target_col)

    feature_cols = [col for col in df_features.columns if col != target_col]
    if target_col not in df_features.columns:
        logger.error(f"Target column '{target_col}' not found in feature-engineered DataFrame.")
        return

    logger.info(f"Feature engineering complete. {len(feature_cols)} features created.")

    # --- 4. Create Sequences ---
    features_data = df_features[feature_cols].values
    target_data = df_features[target_col].values

    X, y = create_sequences(features_data, target_data, SEQ_LEN)

    if len(X) == 0:
        logger.error(f"Not enough data to create sequences with length {SEQ_LEN}. Need more data.")
        return

    # --- 5. Chronological Train/Validation Split (CRITICAL FIX) ---
    split_pct = config.get('train_split_pct', 0.8)
    split_idx = int(len(X) * split_pct)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    logger.info(f"Data split: Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Data split: Validation shapes: X={X_val.shape}, y={y_val.shape}")

    # --- 6. Fit Scalers *Only* on Training Data (CRITICAL FIX) ---
    # Reshape 3D feature data to 2D for fitting the scaler
    n_samples_train, seq_len_train, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples_train * seq_len_train, n_features)

    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train_reshaped)
    logger.info("Feature scaler fitted on training data.")

    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, 1))
    logger.info("Target scaler fitted on training data.")

    # --- 7. Transform All Datasets ---
    # Transform features
    X_train_scaled = feature_scaler.transform(X_train_reshaped).reshape(n_samples_train, seq_len_train, n_features)

    n_samples_val, seq_len_val, _ = X_val.shape
    X_val_reshaped = X_val.reshape(n_samples_val * seq_len_val, n_features)
    X_val_scaled = feature_scaler.transform(X_val_reshaped).reshape(n_samples_val, seq_len_val, n_features)

    # Transform target
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1))

    logger.info("All datasets successfully scaled.")

    # --- 8. Build & Train Model ---
    model = ModelBuilder.build_lstm(seq_len=SEQ_LEN, n_features=n_features)
    logger.info("LSTM model built.")

    model.fit(
        X_train_scaled, y_train_scaled,
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 32),
        validation_data=(X_val_scaled, y_val_scaled),  # (FIX) Use explicit validation set
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ],
        shuffle=True  # Shuffle training sequences (this is good)
    )
    logger.info("Model training complete.")

    # --- 9. Save Artifacts (CRITICAL FIX) ---
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save TF model in H5 format
    model_path = config['model']['model_path_h5']
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save the scalers using joblib
    joblib.dump(feature_scaler, config['model']['feature_scaler_path'])
    logger.info(f"Feature scaler saved to {config['model']['feature_scaler_path']}")

    joblib.dump(target_scaler, config['model']['target_scaler_path'])
    logger.info(f"Target scaler saved to {config['model']['target_scaler_path']}")

    logger.info("--- Pipeline run finished successfully! ---")


if __name__ == "__main__":
    run_pipeline()
