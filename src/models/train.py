import tensorflow as tf
from tensorflow.keras import layers
from .attention import AttentionLayer
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ModelBuilder:
    @staticmethod
    def build_lstm(seq_len, n_features, units=[128, 64, 32], dropout=0.2):
        inputs = layers.Input(shape=(seq_len, n_features))
        x = inputs
        for i, u in enumerate(units[:-1]):
            x = layers.LSTM(u, return_sequences=True)(x)
            x = layers.Dropout(dropout)(x)
        x = layers.LSTM(units[-1], return_sequences=True)(x)
        x = AttentionLayer(units[-1])(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model