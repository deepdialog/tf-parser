
from ..utils.label import Label


def test_label():
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

    l1 = Label(pad=False)
    l1.fit(y1)
    yv1 = l1.transform(y1)
    print(yv1.shape)
    print(yv1)


if __name__ == "__main__":
    test_label()
