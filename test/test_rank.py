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

if __name__ == '__main__':
    unittest.main()

