import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Tokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(X)
        self.vocab_size = len(tokenizer.index_word) + 1
        self.tokenizer = tokenizer

    def transform(self, X):
        return tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(X), padding='post')