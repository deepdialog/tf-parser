
import numpy as np
import tensorflow as tf


class Biaffine(tf.keras.Model):

    def __init__(self, dim0, dim1, odim=1):
        super(Biaffine, self).__init__(self)
        self.dim0 = dim0
        self.dim1 = dim1
        self.w = tf.constant(np.random.rand(odim, dim0, dim1), dtype=tf.float32)

    def call(self, x0, x1):
        assert len(x0.shape) == 3
        assert len(x1.shape) == 3
        assert x0.shape[0] == x1.shape[0]
        assert x0.shape[1] == x1.shape[1]
        assert x0.shape[2] == self.dim0
        assert x1.shape[2] == self.dim1

        x0 = tf.expand_dims(x0, 1)
        x1 = tf.expand_dims(x1, 1)
        r = tf.matmul(x0, self.w)
        r = tf.matmul(
            r,
            tf.transpose(x1, (0, 1, 3, 2))
        )
        if r.shape[1] == 1:
            r = tf.squeeze(r, 1)
        else:
            r = tf.transpose(r, (0, 2, 3, 1))
        return r

