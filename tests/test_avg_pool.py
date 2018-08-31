
import unittest
from src.avg_pool import AvgPool


class TestAvgPool(unittest.TestCase):

    def test_avg_pool(self):
        import numpy as np
        from chainer import Variable
        x = np.ones((2, 2, 1, 3), dtype=np.float32)
        x = Variable(x)

        avg = AvgPool()

        h = avg(x)
        h.grad = np.ones((2, 2, 1), dtype=np.float32)

        h.backward()

        self.assertTrue(np.all(x.grad == np.ones((2, 2, 1, 3), dtype=np.float32) * (1/ np.sqrt(3))))

if __name__ == "__main__":
    unittest.main()