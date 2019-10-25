
from .tokenizer import test_tokenizer
from ..models.embed import Embed


def test_embed():
    v, tokenizer = test_tokenizer()
    emb = Embed(10, tokenizer.vocab_size)
    v = emb(v)
    print(v.shape)
    return v


if __name__ == "__main__":
    test_embed()
