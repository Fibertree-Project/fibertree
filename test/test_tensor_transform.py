import unittest

from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.rank import Rank
from fibertree.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_splitUniform_0(self):
        """ Test splitUniform - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUniform_0.yaml")

        a_out = a.splitUniform(25, depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
        self.assertEqual(a_out.getShape(), [26, 41, 42, 10])


    def test_splitUniform_1(self):
        """ Test splitUniform - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUniform_1.yaml")

        a_out = a.splitUniform(15, depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
        self.assertEqual(a_out.getShape(), [41, 31, 42, 10])


    def test_splitUniform_2(self):
        """ Test splitUniform - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUniform_2.yaml")

        a_out = a.splitUniform(4, depth=2)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
        self.assertEqual(a_out.getShape(), [41, 42, 9, 10])



    def test_splitNonUniform_0(self):
        """ Test splitNonUniform - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_0.yaml")

        a_out = a.splitNonUniform([0, 15, 35], depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
        self.assertEqual(a_out.getShape(), [36, 41, 42, 10])


    def test_splitNonUniform_1(self):
        """ Test splitNonUniform - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_1.yaml")

        a_out = a.splitNonUniform([0, 15, 25], depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
        self.assertEqual(a_out.getShape(), [41, 26, 42, 10])


    def test_splitNonUniform_2(self):
        """ Test splitNonUniform - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_2.yaml")

        a_out = a.splitNonUniform([0, 4, 19], depth=2)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
        self.assertEqual(a_out.getShape(), [41, 42, 5, 10])


    def test_splitEqual_0(self):
        """ Test splitEqual - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_0.yaml")

        a_out = a.splitEqual(2, depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
        self.assertEqual(a_out.getShape(), [41, 41, 42, 10])


    def test_splitEqual_1(self):
        """ Test splitEqual - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_1.yaml")

        a_out = a.splitEqual(2, depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
        self.assertEqual(a_out.getShape(), [41, 34, 42, 10])


    def test_splitEqual_2(self):
        """ Test splitEqual - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_2.yaml")

        a_out = a.splitEqual(2, depth=2)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
        self.assertEqual(a_out.getShape(), [41, 42, 10, 10])



    def test_splitUnEqual_0(self):
        """ Test splitUnEqual - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_0.yaml")

        a_out = a.splitUnEqual([2, 1], depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
        self.assertEqual(a_out.getShape(), [41, 41, 42, 10])



    def test_splitUnEqual_1(self):
        """ Test splitUnEqual - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_1.yaml")

        a_out = a.splitUnEqual([2, 1], depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
        self.assertEqual(a_out.getShape(), [41, 42, 42, 10])


    def test_splitUnEqual_2(self):
        """ Test splitUnEqual - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_2.yaml")

        a_out = a.splitUnEqual([2, 1], depth=2)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
        self.assertEqual(a_out.getShape(), [41, 42, 9, 10])


    def test_swapRanks_0(self):
        """ Test swapRanks - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-swapRanks_0.yaml")

        a_out = a.swapRanks(depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["N", "M", "K"])
        self.assertEqual(a_out.getShape(), [42, 41, 10])


    def test_swapRanks_1(self):
        """ Test swapRanks - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-swapRanks_1.yaml")

        a_out = a.swapRanks(depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "K", "N"])
        self.assertEqual(a_out.getShape(), [41, 10, 42])


    def test_flattenRanks_0(self):
        """ Test flattenRanks - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_out = a.flattenRanks(depth=0)
        a_again = a_out.unflattenRanks(depth=0)

        self.assertEqual(a_again, a)
        self.assertEqual(a_out.getRankIds(), [["M", "N"], "K"])

        # TBD: Semantics for non-integer coordinates
#       self.assertEqual(a_out.getShape(), [7, 10])


    def test_flattenRanks_1(self):
        """ Test flattenRanks - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_out = a.flattenRanks(depth=1)
        a_again = a_out.unflattenRanks(depth=1)

        self.assertEqual(a_again, a)
        self.assertEqual(a_out.getRankIds(), ["M", ["N", "K"]])

        # TBD: Semantics for non-integer coordinates
#       self.assertEqual(a_out.getShape(), [7, 10])


if __name__ == '__main__':
    unittest.main()

