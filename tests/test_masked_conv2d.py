import unittest
import src.masked_conv2d as mc


class TestMaskedConv2d(unittest.TestCase):

    def test_masked_conv(self):
        import numpy as np
        from chainer import Variable

        x = np.ones((1, 1, 6, 6), dtype=np.float32)
        x = Variable(x)
        conv = mc.MaskedConv2D(in_channels=1, out_channels=1, ksize=3, pad=1)

        conv.conv.W.data = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
        conv.conv.b.data *= 0

        t = [[[[2, 4, 4, 4, 4, 4], [3, 6, 6, 6, 6, 6], [3, 6, 6, 6, 6, 6],
               [3, 6, 6, 6, 6, 6], [3, 6, 6, 6, 6, 6], [2, 4, 4, 4, 4, 4]]]]
        t = np.array(t, dtype=np.float32)

        self.assertTrue(np.all(conv(x).data == t))


if __name__ == "__main__":
    unittest.main()
