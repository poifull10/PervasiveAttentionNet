from chainer import Chain
import chainer.functions as F
import numpy as np


class MaxPoolAggreagater(Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        """
        x : shape(B, D, I, J)

        :param x:
        :return: shape(B, D, I)
        """
        return F.max(x, axis=3)


class AvgPoolAggregater(Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        _, _, _, w = x.shape
        return F.sum(x, axis=3) * np.sqrt(1/w)
