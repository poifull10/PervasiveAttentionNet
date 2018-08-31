import unittest
from src.preprocess import *


class TestPreProcess(unittest.TestCase):

    def test_add_bos_remove_eos(self):
        test_string = """<title>hogehoge</title>\nIt can be a very complicated thing, the ocean."""
        out = sentence2token(test_string)
        self.assertEqual(out[0], "<BOS>")
        self.assertEqual(out[1], "It")
        self.assertEqual(out[-1], "<EOS>")
        self.assertEqual(out[-2], "ocean")
        self.assertEqual(len(out), 12)

    def test_split_word(self):
        test_string = """It can be a very complicated thing, the ocean.\n\
        It can be a very complicated thing, the ocean!\n\
        Can it be a very complicated thing, the ocean?"""
        out = sentence2token(test_string)
        self.assertEqual(out[8], ",")
        self.assertEqual(out[11], "<EOS>")
        self.assertEqual(out[23], "!")
        self.assertEqual(out[36], "?")
        self.assertEqual(len(out), 38)




if __name__ == "__main__":
    unittest.main()