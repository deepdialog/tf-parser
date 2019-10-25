import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .models.parser_model import ParserModel
from .utils.label import Label
from .utils.tokenizer import Tokenizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TFParser:
    def __init__(self, embedding_size=100, batch_size=32, epoch=100):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = None

    def build_model(self):
        return ParserModel(embedding_size=self.embedding_size,
                           vocab_size=self.tokenizer.vocab_size,
                           proj0_size=200,
                           proj1_size=200,
                           tag0_size=self.label0.label_size,
                           tag1_size=self.label1.label_size)

    def fit(self, X, y0, y1):
        """Model training."""

        tokenizer = Tokenizer()
        tokenizer.fit(X)
        self.tokenizer = tokenizer

        label0 = Label()
        label0.fit(y0)
        self.label0 = label0

        label1 = Label()
        label1.fit(y1)
        self.label1 = label1

        if self.model is None:
            model = self.build_model()
            self.model = model
        else:
            model = self.model

        optimizer = tf.keras.optimizers.Adam()

        # X_vec = tokenizer.transform(X)
        # y0_vec = label0.transform(y0)
        # y1_vec = label1.transform(y1)

        total_batch = int(np.ceil(len(X) / self.batch_size))
        for i_epoch in range(self.epoch):
            pbar = tqdm(range(total_batch), ncols=100)
            pbar.set_description(f'epoch: {i_epoch} loss: /')
            losses = []
            for i in pbar:
                i_min = i * self.batch_size
                i_max = min((i + 1) * self.batch_size, len(X))
                x = tokenizer.transform(X[i_min:i_max])
                y0b = label0.transform(y0[i_min:i_max])
                y1b = label1.transform(y1[i_min:i_max])
                x = tf.convert_to_tensor(x, dtype=tf.int32)
                y0b = tf.convert_to_tensor(y0b, dtype=tf.int32)
                y1b = tf.convert_to_tensor(y1b, dtype=tf.int32)
                with tf.GradientTape() as tape:
                    loss = model.compute_loss(x, y0b, y1b)
                    gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                loss = loss.numpy().sum()
                losses.append(loss)

                pbar.set_description(
                    f'epoch: {i_epoch} loss: {np.mean(losses):.4f}')

    def predict(self, X):
        """Predict label."""
        assert self.model is not None, 'Intent not fit'
        x = self.tokenizer.transform(X)
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        r0, r1 = self.model(x)
        r0 = tf.argmax(r0, -1)
        r1 = tf.argmax(r1, -1)
        r0 = r0.numpy()
        r1 = r1.numpy()
        r = [[
            self.label1.inverse_transform(rrr).tolist()[:len(X[i])]
            for rrr in rr
        ][:len(X[i])] for i, rr in enumerate(r1)]
        return r

    def __getstate__(self):
        """Pickle compatible."""
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_weights'] = state['model'].get_weights()
            del state['model']
        return state

    def __setstate__(self, state):
        """Pickle compatible."""
        if 'model_weights' in state:
            model_weights = state.get('model_weights')
            del state['model_weights']
            self.__dict__.update(state)
            self.model = self.build_model()
            self.model.set_weights(model_weights)
        else:
            self.__dict__.update(state)
