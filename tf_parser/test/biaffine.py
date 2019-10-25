import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from .encoder import test_encoder
from ..models.biaffine import Biaffine


def test_biaffine():
    v = test_encoder()
    bia0 = Biaffine(20, 20, 7)
    v0 = bia0(v, v)
    print(v0.shape)
    bia1 = Biaffine(20, 20)
    v1 = bia1(v, v)
    print(v1.shape)


if __name__ == "__main__":
    test_biaffine()
