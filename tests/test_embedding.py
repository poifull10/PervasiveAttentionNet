import unittest
from src.embedding import Embedding


class TestEmbedding(unittest.TestCase):

    def test_embedding(self):
        import numpy as np

        input_seq_len = 3
        output_seq_len = 4
        ds = 10
        dt = 20

        src_seq = np.arange(input_seq_len).reshape(-1, input_seq_len).astype(np.int32)
        dst_seq = output_seq_len - 1 - np.arange(output_seq_len).reshape(-1, output_seq_len).astype(np.int32)

        emb = Embedding(vocab_src_size=input_seq_len,
                        vocab_dst_size=output_seq_len,
                        bad_word=-1, ds=ds, dt=dt)

        out = emb(src_seq, dst_seq)

        self.assertTupleEqual(out.shape, (1, ds + dt, input_seq_len, output_seq_len))

if __name__ == "__main__":
    unittest.main()