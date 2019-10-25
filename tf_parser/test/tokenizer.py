
from ..utils.tokenizer import Tokenizer


def test_tokenizer():
    x = [
        ['我', '爱', '你'],
        ['我', '爱', '你'],
    ]
    tokenizer = Tokenizer()
    tokenizer.fit(x)
    v = tokenizer.transform(x)
    print(v.shape)
    print(v)
    return v, tokenizer


if __name__ == "__main__":
    test_tokenizer()
