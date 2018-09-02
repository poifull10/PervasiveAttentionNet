import unittest
from src.loss import seq_cross_entropy
from chainer import Variable


class TestSeqCrossEntropy(unittest.TestCase):
    def test_seq_cross_entropy(self):
        import numpy as np

        dst_seq = np.array([[2, 10, 3, 1, 2, -1, -1]], dtype=np.int32)
        p = Variable(
            np.array([[1.0, 1.0, 0., 0., 0., 1, 1]], dtype=np.float32))
        self.assertEqual(seq_cross_entropy(p, dst_seq, 1e-10).data, 0)

        dst_seq = np.array([[2, 10, 3, 1, 2, -1, -1]], dtype=np.int32)
        p = Variable(
            np.array([[0.5, 1.0, 0.5, 0., 0., 0.5, 0.3]], dtype=np.float32))
        tole = 1e-5
        self.assertTrue(seq_cross_entropy(
            p, dst_seq, 1e-10).data < - 2 * 0.5 * np.log(0.5) + tole)
        self.assertTrue(seq_cross_entropy(
            p, dst_seq, 1e-10).data > - 2 * 0.5 * np.log(0.5) - tole)


if __name__ == "__main__":
    unittest.main()
