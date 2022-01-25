import unittest
from fibertree import Payload
from fibertree import Fiber
from fibertree import Rank


class TestRank(unittest.TestCase):

    def test_default_format(self):
        rank = Rank("K")
        self.assertEqual(rank.getFormat(), "C")

    def test_set_format(self):
        rank = Rank("K")

        rank.setFormat("U")
        self.assertEqual(rank.getFormat(), "U")

        rank.setFormat("C")
        self.assertEqual(rank.getFormat(), "C")

        self.assertRaises(AssertionError, lambda: rank.setFormat("G"))

    def test_set_collecting(self):
        rank = Rank("K")
        self.assertFalse(rank.getCollecting())

        rank.setCollecting(True)
        self.assertTrue(rank.getCollecting())

        self.assertRaises(AssertionError, lambda: rank.setCollecting("foo"))

if __name__ == '__main__':
    unittest.main()

