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

    def test_get_attrs(self):
        rank = Rank("K", shape=20)
        rank.setFormat("U").setDefault(3).setId("M")

        attrs = RankAttrs("K", shape=20)
        attrs.setFormat("U").setDefault(3).setId("M")

        self.assertEqual(rank.getAttrs(), attrs)

    def test_append_pop(self):
        f1 = lambda: Fiber([1, 2], [3, 4])
        f2 = lambda: Fiber([5, 6], [7, 8])
        self.assertIsNone(f1().getOwner())
        self.assertIsNone(f2().getOwner())

        rank = Rank("K")
        rank.append(f1())
        rank.append(f2())

        self.assertEqual(rank.getFibers(), [f1(), f2()])
        self.assertIsNotNone(rank.getFibers()[0].getOwner())
        self.assertIsNotNone(rank.getFibers()[1].getOwner())

        f2_final = rank.pop()

        self.assertEqual(f2_final, f2())
        self.assertIsNone(f2_final.getOwner())
        self.assertIsNotNone(rank.getFibers()[0].getOwner())

if __name__ == '__main__':
    unittest.main()

