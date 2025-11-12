import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class DataTransformer:
    def __init__(self):
        self.scaler = StandardScaler()

    #old method
    # def fit_scaler(self, data):
    #     return self.scaler.fit_transform(data.reshape(-1, 1))

    #new method
    def fit_scalers(self, X_train, y_train):
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train.reshape(-1, 1))

    def create_sequences(self, data, seq_len=168):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)