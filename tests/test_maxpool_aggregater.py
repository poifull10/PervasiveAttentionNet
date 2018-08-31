import unittest
from src.aggregater import MaxPoolAggreagater
from chainer import Variable


class TestMaxPoolAggregater(unittest.TestCase):

    def test_max_pool(self):
        import numpy as np
        x = np.arange(2*2*3*3, dtype=np.float32).reshape(2, 2, 3, 3)
        y = Variable(x)
        z = Variable(np.array([[[2, 5, 8],
                               [11, 14, 17]],
                               [[20, 23, 26],
                               [29, 32, 35]]], dtype=np.float32))
        agg = MaxPoolAggreagater()
        y = agg(y)
        self.assertTrue(np.all(y.data == z.data))




if __name__ == "__main__":
    unittest.main()