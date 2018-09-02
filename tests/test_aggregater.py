import unittest
from src.aggregater import MaxPoolAggreagater, AvgPoolAggregater
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


class TestAvgPoolAggregater(unittest.TestCase):

    def test_avg_pool(self):
        import numpy as np
        from chainer import Variable
        x = np.ones((2, 2, 1, 3), dtype=np.float32)
        x = Variable(x)

        avg = AvgPoolAggregater()

        h = avg(x)
        h.grad = np.ones((2, 2, 1), dtype=np.float32)

        h.backward()

        one = np.ones((2, 2, 1, 3), dtype=np.float32)
        self.assertTrue(np.all(x.grad == one * (1 / np.sqrt(3))))

if __name__ == "__main__":
    unittest.main()
