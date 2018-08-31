from chainer import Chain
from chainer.function_node import FunctionNode
import chainer.functions as F
import numpy as np

class AvgPool(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            pass

    def __call__(self, x):
        _, _, _, w = x.shape
        return F.sum(x, axis=3) * np.sqrt(1/w)
