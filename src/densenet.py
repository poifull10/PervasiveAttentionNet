from chainer import Chain
from chainer.links import BatchNormalization, Convolution2D
import chainer.functions as F
from src.masked_conv2d import MaskedConv2D


class ConvBlock(Chain):
    def __init__(self, k, layer_num, f0, growth=4, dropout_ratio=0.5):
        super().__init__()
        with self.init_scope():
            self.bn1 = BatchNormalization(size=(f0 + (layer_num-1)*growth))
            self.bn2 = BatchNormalization(size=4 * growth)
            self.conv1 = Convolution2D(in_channels=f0 + (layer_num-1)*growth,
                                       out_channels=4 * growth,
                                       ksize=1)
            self.conv2 = MaskedConv2D(in_channels=4 * growth,
                                      out_channels=growth,
                                      ksize=k,
                                      pad=k//2)
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.dropout(x, self.dropout_ratio)
        return x


class DenseNet(Chain):
    def __init__(self, block_num, k, ds, dt, growth=4):
        super().__init__()
        with self.init_scope():
            self.conv_blocks = [
                    ConvBlock(k, i+1, f0=ds+dt, growth=growth)
                    for i in range(block_num)]

    def __call__(self, x):
        H = x
        for layer in self.conv_blocks:
            G = layer(H)
            H = F.concat([H, G], axis=1)
        return H
