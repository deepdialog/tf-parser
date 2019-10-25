import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from ..models.parser_model import ParserModel
from .tokenizer import test_tokenizer


def test():
    v, tokenizer = test_tokenizer()
    pm = ParserModel(100, tokenizer.vocab_size, 50, 50, 2, 15)
    r0, r1 = pm(v)
    print(r0.shape)
    print(r1.shape)

    y0 = tf.random.uniform(
        (2, 3, 3),
        minval=0,
        maxval=2,
        dtype=tf.dtypes.int32
    )
    y1 = tf.random.uniform(
        (2, 3, 3),
        minval=0,
        maxval=15,
        dtype=tf.dtypes.int32
    )

    loss = pm.compute_loss(v, y0, y1)
    print(loss)


if __name__ == "__main__":
    test()