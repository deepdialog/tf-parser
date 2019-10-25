import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from .embed import test_embed
from ..models.encoder import Encoder


def test_encoder():
    v = test_embed()
    print(v.shape)
    en = Encoder(10)
    v = en(v)
    print(v.shape)
    return v


if __name__ == "__main__":
    test_encoder()
