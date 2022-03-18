import unittest
from fibertree import Payload
from fibertree import Fiber
from fibertree import Rank
from fibertree.core.rank_attrs import RankAttrs


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

    def test_get_attrs(self):
        rank = Rank("K", shape=20)
        rank.setFormat("U").setDefault(3).setId("M").setCollecting(True)

        attrs = RankAttrs("K", shape=20)
        attrs.setFormat("U").setDefault(3).setId("M").setCollecting(True)

        self.assertEqual(rank.getAttrs(), attrs)

if __name__ == '__main__':
    unittest.main()

