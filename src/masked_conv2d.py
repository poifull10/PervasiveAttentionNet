
from chainer import Chain, Variable
from chainer.links import BatchNormalization, Convolution2D
import chainer.functions as F
import numpy as np

class MaskedConv2D(Chain):
    def __init__(self, in_channels, out_channels, ksize, pad):
        super().__init__()
        with self.init_scope():
            self.conv = Convolution2D(in_channels=in_channels, out_channels= out_channels,
                                      ksize=ksize, pad=pad)
        _, _, h, w = self.conv.W.shape
        _mask = np.ones(shape=(h, w))
        _mask[:, w//2+1:] = 0
        self.mask = Variable(_mask)


    def __call__(self, x):
        self.conv.W.data[:, :] *= self.mask.data
        return self.conv(x)

if __name__ == "__main__":
    x = np.ones((1, 1, 10, 10), dtype=np.float32)
    x = Variable(x)
    conv = MaskedConv2D(in_channels=1, out_channels=1, ksize=3, pad=1)
    print(conv(x))