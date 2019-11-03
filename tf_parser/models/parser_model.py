
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .embed import Embed
from .encoder import Encoder
from .biaffine import Biaffine

class ParserModel(tf.keras.Model):
    def __init__(self,
                 embedding_size,
                 vocab_size,
                 proj0_size,
                 proj1_size,
                 tag0_size,
                 tag1_size):
        super(ParserModel, self).__init__(self)
        self.emb = Embed(
            embedding_size=embedding_size,
            vocab_size=vocab_size
        )
        self.en = Encoder(
            embedding_size=hidden_size,
            hidden_size=embedding_size
        )

        self.p0 = tf.keras.models.Sequential([
            tf.keras.layers.Dense(proj0_size)
        ])
        self.p0.build(input_shape=(None, embedding_size * 2))

        self.p1 = tf.keras.models.Sequential([
            tf.keras.layers.Dense(proj1_size)
        ])
        self.p1.build(input_shape=(None, embedding_size * 2))

        self.b0 = Biaffine(proj0_size, proj1_size, tag0_size)
        self.b1 = Biaffine(proj0_size, proj1_size, tag1_size)

    def call(self, inputs):
        """
        input:
            inputs: [batch_size, lengths]
        """
        lengths = tf.reduce_sum(tf.cast(tf.math.greater(inputs, 0), tf.int32), axis=-1)
        # m: [batch_size, lengths, embedding_size]
        m = self.emb(inputs)
        # m: [batch_size, lengths, hidden_size]
        m = self.en(m)
        # x0: [batch_size, lengths, proj0_size]
        x0 = self.p0(m)
        # x1: [batch_size, lengths, proj1_size]
        x1 = self.p1(m)
        # t0: [batch_size, lengths, lengths, tag0_size]
        t0 = self.b0(x0, x1)
        # t1: [batch_size, lengths, lengths, tag1_size]
        t1 = self.b1(x0, x1)
        return t0, t1

    def compute_loss(self, inputs, y0, y1):
        r0, r1 = self.call(inputs)
        mask = tf.cast(tf.math.greater(y0, 0), tf.int32)
        mask = tf.cast(mask, tf.float32)

        r0 = r0 * tf.tile(tf.expand_dims(mask, -1), (1, 1, 1, r0.shape[-1]))
        r1 = r1 * tf.tile(tf.expand_dims(mask, -1), (1, 1, 1, r1.shape[-1]))

        r0 = tf.reshape(r0, (-1, r0.shape[-1]))
        r1 = tf.reshape(r1, (-1, r1.shape[-1]))

        y0 = tf.reshape(y0, (-1,))
        y1 = tf.reshape(y1, (-1,))

        l0 = tf.nn.sparse_softmax_cross_entropy_with_logits(y0, r0)
        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(y1, r1)

        loss = tf.reduce_mean(l0) + tf.reduce_mean(l1)
        return loss
