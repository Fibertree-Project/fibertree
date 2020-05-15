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
        

    def test_splitUniform_1(self):
        """ Test splitUniform - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUniform_1.yaml")

        a_out = a.splitUniform(15, depth=1)

        self.assertEqual(a_out, a_verify)
        

    def test_splitUniform_2(self):
        """ Test splitUniform - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUniform_2.yaml")

        a_out = a.splitUniform(4, depth=2)

        self.assertEqual(a_out, a_verify)


    def test_splitNonUniform_0(self):
        """ Test splitNonUniform - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_0.yaml")

        a_out = a.splitNonUniform([0, 15, 35], depth=0)

        self.assertEqual(a_out, a_verify)


    def test_splitNonUniform_1(self):
        """ Test splitNonUniform - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_1.yaml")

        a_out = a.splitNonUniform([0, 15, 25], depth=1)

        self.assertEqual(a_out, a_verify)


    def test_splitNonUniform_2(self):
        """ Test splitNonUniform - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitNonUniform_2.yaml")

        a_out = a.splitNonUniform([0, 4, 19], depth=2)

        self.assertEqual(a_out, a_verify)


    def test_splitEqual_0(self):
        """ Test splitEqual - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_0.yaml")

        a_out = a.splitEqual(2, depth=0)

        self.assertEqual(a_out, a_verify)


    def test_splitEqual_1(self):
        """ Test splitEqual - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_1.yaml")

        a_out = a.splitEqual(2, depth=1)

        self.assertEqual(a_out, a_verify)


    def test_splitEqual_2(self):
        """ Test splitEqual - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitEqual_2.yaml")

        a_out = a.splitEqual(2, depth=2)

        self.assertEqual(a_out, a_verify)



    def test_splitUnEqual_0(self):
        """ Test splitUnEqual - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_0.yaml")

        a_out = a.splitUnEqual([2, 1], depth=0)

        self.assertEqual(a_out, a_verify)



    def test_splitUnEqual_1(self):
        """ Test splitUnEqual - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_1.yaml")

        a_out = a.splitUnEqual([2, 1], depth=1)

        self.assertEqual(a_out, a_verify)


    def test_splitUnEqual_2(self):
        """ Test splitUnEqual - depth=2 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-splitUnEqual_2.yaml")

        a_out = a.splitUnEqual([2, 1], depth=2)

        self.assertEqual(a_out, a_verify)


    def test_swapRanks_0(self):
        """ Test swapRanks - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-swapRanks_0.yaml")

        a_out = a.swapRanks(depth=0)

        self.assertEqual(a_out, a_verify)


    def test_swapRanks_1(self):
        """ Test swapRanks - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_verify = Tensor("./data/tensor_transform-a-swapRanks_1.yaml")

        a_out = a.swapRanks(depth=1)

        self.assertEqual(a_out, a_verify)


    def test_flattenRanks_0(self):
        """ Test flattenRanks - depth=0 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_out = a.flattenRanks(depth=0)
        a_again = a_out.unflattenRanks(depth=0)

        self.assertEqual(a_again, a)


    def test_flattenRanks_1(self):
        """ Test flattenRanks - depth=1 """

        a = Tensor("./data/tensor_transform-a.yaml")
        a_out = a.flattenRanks(depth=1)
        a_again = a_out.unflattenRanks(depth=1)

        self.assertEqual(a_again, a)


if __name__ == '__main__':
    unittest.main()

