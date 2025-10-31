import tensorflow as tf
from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')
        self.u = self.add_weight(shape=(self.units,), initializer='glorot_uniform')

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        context = inputs * attention_weights
        return tf.reduce_sum(context, axis=1)

    def get_config(self):
        return {**super().get_config(), "units": self.units}