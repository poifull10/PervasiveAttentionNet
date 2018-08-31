
from chainer import Chain, Variable
from chainer.links import BatchNormalization, Convolution2D
import chainer.functions as F

class MaxPoolAggreagater(Chain):
    def __init__(self):
        pass

    def __call__(self, x):
        """
        x : shape(B, D, I, J)

        :param x:
        :return: shape(B, D, I)
        """
        return F.max(x, axis=3)
