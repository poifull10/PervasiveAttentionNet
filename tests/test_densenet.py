import unittest
from src.densenet import DenseNet


class TestDenseNet(unittest.TestCase):

    def test_densenet(self):
        import numpy as np

        ds = 10
        dt = 20
        layer= 7
        g = 4
        height = 10
        width = 10

        input = np.ones((1, ds+dt, height, width)).astype(np.float32)
        densenet = DenseNet(block_num=layer, k=3, ds=ds, dt=dt, growth=g)
        output = densenet(input)

        self.assertTupleEqual(output.shape, (1, ds + dt + layer*g, height, width))

if __name__ == "__main__":
    unittest.main()