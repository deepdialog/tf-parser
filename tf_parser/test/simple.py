
from ..utils.label import Label


def test_label():
    x = [
        ['我', '爱', '你'],
        ['我', '爱', '你'],
    ]
    y0 = [
        [
            ['link', 'link', 'link'],
            ['link', 'link', 'link'],
            ['no_link', 'no_link', 'no_link']
        ],
        [
            ['link', 'link', 'link'],
            ['link', 'link', 'link'],
            ['no_link', 'no_link', 'no_link']
        ]
    ]
    y1 = [
        [
            ['obj', 'obj', 'obj'],
            ['dobj', 'dobj', 'dobj'],
            ['sub', 'sub', 'sub']
        ],
        [
            ['obj', 'obj', 'obj'],
            ['dobj', 'dobj', 'dobj'],
            ['sub', 'sub', 'sub']
        ]
    ]
    l0 = Label(pad=False)
    l0.fit(y0)
    yv0 = l0.transform(y0)
    print(yv0.shape)
    print(yv0)


if __name__ == "__main__":
    test_label()
