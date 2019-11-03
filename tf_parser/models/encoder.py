
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Encoder(tf.keras.Model):

    def __init__(self, embedding_size, hidden_size):
        super(Encoder, self).__init__(self)
        inputs = tf.keras.layers.Input(
            shape=(None, embedding_size,),
            dtype=tf.float32)
        x = inputs
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        )(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs):
        """
        input: [B, L, E]
        output: [B, L, H]
        """
        return self.model(inputs)
