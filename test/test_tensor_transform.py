import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Rank
from fibertree import Tensor


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

    def test_flattenRanks_f01(self):
        """ Test flattenRanks - f01 """

        t0 = Tensor.fromYAMLfile("./data/tensor_3d-0.yaml")

        f01 = t0.flattenRanks(depth=0, levels=1)
        u01 = f01.unflattenRanks(depth=0, levels=1)
        
        self.assertEqual(u01, t0)

    def test_flattenRanks_f02(self):
        """ Test flattenRanks - f02 """

        t0 = Tensor.fromYAMLfile("./data/tensor_3d-0.yaml")

        f02 = t0.flattenRanks(depth=0, levels=2)
        u02a = f02.unflattenRanks(depth=0, levels=1)
        u02b = u02a.unflattenRanks(depth=1, levels=1)

        self.assertEqual(u02b, t0)

        u02 = f02.unflattenRanks(depth=0, levels=2)

        self.assertEqual(u02, t0)

    def test_flattenRanks_f12(self):
        """ Test flattenRanks - f12 """

        t0 = Tensor.fromYAMLfile("./data/tensor_3d-0.yaml")

        f12 = t0.flattenRanks(depth=1, levels=1)
        u12 = f12.unflattenRanks(depth=1, levels=1)
        self.assertEqual(u12, t0)


    def test_flattenRanks_f02(self):
        """ Test flattenRanks - f02 """

        t0 = Tensor.fromYAMLfile("./data/tensor_3d-0.yaml")
        t1 = Tensor.fromYAMLfile("./data/tensor_3d-1.yaml")

        t2 = Tensor.fromFiber(["A", "B", "C", "D"], 
                              Fiber([1, 4], [t0.getRoot(), t1.getRoot()]),
                              name="t2")

        f13 = t2.flattenRanks(depth=1, levels=2)
        u13 = f13.unflattenRanks(depth=1, levels=2)

        self.assertEqual(u13, t2)

        f04 = t2.flattenRanks(depth=0, levels=3)
        u04 = f04.unflattenRanks(depth=0, levels=3)

        self.assertEqual(u04, t2)


if __name__ == '__main__':
    unittest.main()

