
from ..tf_parser import TFParser

tfp = TFParser(epoch=10, batch_size=64)

import pickle

print('load data')
x, y0, y1 = pickle.load(open('./data.pkl', 'rb'))

x = x
x = [['<S>'] + xx for xx in x]
y1 = y1
y0 = [
    [
        [
            'LINK' if item != 'NONE' else item
            for item in row
        ]
        for row in y
    ]
    for y in y1
]
print('loaded')

tfp.fit(x, y0, y1)

r = tfp.predict(x[:3])
print(r)
