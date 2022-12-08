"""Tests related to shape of a tensor"""

import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Metrics
from fibertree import Rank
from fibertree import Tensor


class TestTensorShape(unittest.TestCase):

    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

    def test_shape_empty(self):
        """Test shape of empty tensor"""

        t1 = Tensor(rank_ids=["M", "K"])

        self.assertEqual(t1.getRankIds(), ["M", "K"])
        self.assertEqual(t1.getShape(), [0, 0])

        t2 = Tensor(rank_ids=["M", "K"], shape=[10,20])

        self.assertEqual(t2.getRankIds(), ["M", "K"])
        self.assertEqual(t2.getShape(), [10, 20])

    def test_shape_0D(self):
        """Test shpe of 0-D tensor"""

        t = Tensor(rank_ids=[])
        p = t.getRoot()
        p += 1

        self.assertEqual(t.getRankIds(), [])
        self.assertEqual(t.getShape(), [])

    def test_shape_new(self):
        """Test shape of a tensor from a file"""

        t1 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        self.assertEqual(t1.getRankIds(), ["M", "K"])
        self.assertEqual(t1.getShape(), [7, 4])


        # Note: We cannot override the shape of shape from a YAML file

    def test_setShape(self):
        """Test setShape"""
        t1 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        self.assertEqual(t1.getShape(), [7, 4])

        t1.setShape([100, 200])
        self.assertEqual(t1.getShape(), [100, 200])


    def test_shape_fromUncompressed_1D(self):
        """Test shape of a tensor from 1D nested lists"""

        l1 = [ 100, 101, 0, 102 ]

        t1 = Tensor.fromUncompressed(["M"], l1)

        self.assertEqual(t1.getRankIds(), ["M"])
        self.assertEqual(t1.getShape(), [ 4 ])

        l2 = [ 100, 101, 0, 0 ]

        t2 = Tensor.fromUncompressed(["M"], l2)

        self.assertEqual(t2.getRankIds(), ["M"])
        self.assertEqual(t2.getShape(), [ 4 ])

    def test_shape_fromUncompressed_2D_A1(self):
        """Test shape of a tensor from 2D nested lists (tensor A)"""

        #         0    1    2    3
        #
        l1 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0, 203 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0, 603 ] ] # 6


        t1 = Tensor.fromUncompressed(["M", "K"], l1)

        with self.subTest(test="All ranks"):
            self.assertEqual(t1.getRankIds(), ["M", "K"])
            self.assertEqual(t1.getShape(), [ 7, 4 ])

        with self.subTest(test="All ranks specified"):
            self.assertEqual(t1.getShape(["M", "K"]), [7, 4])

        with self.subTest(test="Just rank 'M' as list"):
            self.assertEqual(t1.getShape(["M"]), [7])

        with self.subTest(test="Just rank 'K' as list"):
            self.assertEqual(t1.getShape(["K"]), [4])

        with self.subTest(test="Just rank 'M'"):
            self.assertEqual(t1.getShape("M"), 7)

        with self.subTest(test="Just rank 'K'"):
            self.assertEqual(t1.getShape("K"), 4)

        with self.subTest(test="Check authoritative"):
            self.assertEqual(t1.getShape(authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["M", "K"], authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["M"], authoritative=True), [7])
            self.assertEqual(t1.getShape(["K"], authoritative=True), [4])
            self.assertEqual(t1.getShape("M", authoritative=True), 7)
            self.assertEqual(t1.getShape("K", authoritative=True), 4)


    def test_shape_fromUncompressed_2D_A2(self):
        """Test shape of a tensor from 2D nested lists (tensor A, multiletter ranks_ids)"""

        #         0    1    2    3
        #
        l1 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0, 203 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0, 603 ] ] # 6


        t1 = Tensor.fromUncompressed(["MA", "KA"], l1)

        with self.subTest(test="All ranks"):
            self.assertEqual(t1.getRankIds(), ["MA", "KA"])
            self.assertEqual(t1.getShape(), [ 7, 4 ])

        with self.subTest(test="All ranks specified"):
            self.assertEqual(t1.getShape(["MA", "KA"]), [7, 4])

        with self.subTest(test="Just rank 'MA' as list"):
            self.assertEqual(t1.getShape(["MA"]), [7])

        with self.subTest(test="Just rank 'KA' as list"):
            self.assertEqual(t1.getShape(["KA"]), [4])

        with self.subTest(test="Just rank 'MA'"):
            self.assertEqual(t1.getShape("MA"), 7)

        with self.subTest(test="Just rank 'KA'"):
            self.assertEqual(t1.getShape("KA"), 4)

        with self.subTest(test="Check authoritative"):
            self.assertEqual(t1.getShape(authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["MA", "KA"], authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["MA"], authoritative=True), [7])
            self.assertEqual(t1.getShape(["KA"], authoritative=True), [4])
            self.assertEqual(t1.getShape("MA", authoritative=True), 7)
            self.assertEqual(t1.getShape("KA", authoritative=True), 4)


    def test_shape_fromUncompressed_2D_B(self):
        """Test shape of a tensor from 2D nested lists (tensor B)"""

        #         0    1    2    3
        #
        l2 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0,   0 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0,   0 ] ] # 6

        t2 = Tensor.fromUncompressed(["M", "K"], l2)

        self.assertEqual(t2.getRankIds(), ["M", "K"])
        self.assertEqual(t2.getShape(), [7, 4])


    def test_shape_fromFiber(self):
        """Test shape of a tensor from a fiber without authoritative shape"""

        y1 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        f1 = y1.getRoot()

        t1 = Tensor.fromFiber(["M", "K"], f1)

        with self.subTest(test="All ranks"):
            self.assertEqual(t1.getRankIds(), ["M", "K"])
            self.assertEqual(t1.getShape(), [7, 4])

        with self.subTest(test="All ranks specified"):
            self.assertEqual(t1.getShape(["M", "K"]), [7, 4])

        with self.subTest(test="Just rank 'M' as list"):
            self.assertEqual(t1.getShape(["M"]), [7])

        with self.subTest(test="Just rank 'K' as list"):
            self.assertEqual(t1.getShape(["K"]), [4])

        with self.subTest(test="Just rank 'M'"):
            self.assertEqual(t1.getShape("M"), 7)

        with self.subTest(test="Just rank 'K'"):
            self.assertEqual(t1.getShape("K"), 4)

        with self.subTest(test="Check authoritative"):
            self.assertEqual(t1.getShape(authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["M", "K"], authoritative=True), [7, 4])
            self.assertEqual(t1.getShape(["M"], authoritative=True), [7])
            self.assertEqual(t1.getShape(["K"], authoritative=True), [4])
            self.assertEqual(t1.getShape("M", authoritative=True), 7)
            self.assertEqual(t1.getShape("K", authoritative=True), 4)


    def test_shape_fromFiber_authoritative(self):
        """Test shape of a tensor from a fiber with authoritative shape"""

        y1 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        f1 = y1.getRoot()
        t1 = Tensor.fromFiber(["M", "K"], f1, [100,200])

        with self.subTest(test="All ranks"):
            self.assertEqual(t1.getRankIds(), ["M", "K"])
            self.assertEqual(t1.getShape(), [100, 200])

        with self.subTest(test="All ranks specified"):
            self.assertEqual(t1.getShape(["M", "K"]), [100, 200])

        with self.subTest(test="Just rank 'M'"):
            self.assertEqual(t1.getShape(["M"]), [100])

        with self.subTest(test="Just rank 'K'"):
            self.assertEqual(t1.getShape(["K"]), [200])

        with self.subTest(test="Just rank 'M'"):
            self.assertEqual(t1.getShape("M"), 100)

        with self.subTest(test="Just rank 'K'"):
            self.assertEqual(t1.getShape("K"), 200)

        with self.subTest(test="Check authoritative"):
            self.assertEqual(t1.getShape(authoritative=True), [100, 200])
            self.assertEqual(t1.getShape(["M", "K"], authoritative=True), [100, 200])
            self.assertEqual(t1.getShape(["M"], authoritative=True), [100])
            self.assertEqual(t1.getShape(["K"], authoritative=True), [200])
            self.assertEqual(t1.getShape("M", authoritative=True), 100)
            self.assertEqual(t1.getShape("K", authoritative=True), 200)

    def test_shape_after_populate(self):
        """Test shape after populate"""
        Z_M = Tensor(rank_ids=["M"])
        A_M = Tensor.fromUncompressed(rank_ids=["M"], root=[1, 2, 3, 4])

        for m, (z_ref, a_val) in Z_M.getRoot() << A_M.getRoot():
            z_ref += a_val

        self.assertEqual(Z_M.getShape(), [4])

    def test_rankid_2D(self):
        """Test setting rank ids of 2D tensor"""

        #         0    1    2    3
        #
        l1 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0, 203 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0, 603 ] ] # 6


        rank_ids = ["M", "K"]
        t1 = Tensor.fromUncompressed(rank_ids, l1)

        rank_ids2 = t1.getRankIds()

        self.assertEqual(rank_ids2, rank_ids)

        rank_ids_new = ["M2", "M1"]
        t1.setRankIds(rank_ids_new)

        rank_ids3 = t1.getRankIds()

        self.assertEqual(rank_ids3, rank_ids_new)


if __name__ == '__main__':
    unittest.main()

