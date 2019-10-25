import os

import numpy as np
from sklearn.preprocessing import LabelEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


PAD = '<PAD>'


class Label:
    def __init__(self, pad=True):
        self.pad = pad

    def fit(self, y):
        label = LabelEncoder()
        tags = []
        for yy in y:
            if isinstance(yy, (list, tuple)):
                for yyy in yy:
                    tags += yyy
            else:
                tags += yy
        if self.pad:
            label.fit([PAD] + tags)
        else:
            label.fit(tags)
        self.label = label
        label_size = len(label.classes_)
        assert label_size >= 2
        self.label_size = label_size

    def transform(self, y):
        max_length = int(np.max([len(yy) for yy in y]))
        if isinstance(y[0], (list, tuple)):
            # 3D
            pad_row = self.label.transform([PAD] * max_length)
            return np.array([
                [
                    self.label.transform(yyy + [PAD] * (max_length - len(yyy))) if self.pad else self.label.transform(yyy)
                    for yyy in yy
                ] + ([pad_row] * (max_length - len(yy)))
                for yy in y
            ])
        else:
            # 2D
            return np.array([
                self.label.transform(yy + [PAD] * (max_length - len(yy))) if self.pad else self.label.transform(yy)
                for yy in y
            ])

    def inverse_transform(self, y):
        return self.label.inverse_transform(y)