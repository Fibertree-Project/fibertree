import unittest

from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.rank import Rank
from fibertree.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_constructor(self):
        """Test construction of empty tensor"""

        Tensor(rank_ids=["M", "K"])


    def test_new(self):
        """Test construction of a tensor from a file"""

        Tensor("./data/test_tensor-1.yaml")

    def test_equal(self):
        """Test equality comparison"""

        tensor1 = Tensor("./data/test_tensor-1.yaml")
        tensor2 = Tensor("./data/test_tensor-1.yaml")

        self.assertTrue(tensor1 == tensor2)

    def test_fromYAML(self):
        """Test construction from a YAML file"""

        tensor_ref  = Tensor("./data/test_tensor-1.yaml")

        tensor = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        self.assertTrue(tensor == tensor_ref)

    def test_fromUncompressed_1D(self):
        """Test construction of a tensor from nested lists"""

        tensor_ref = Tensor("./data/test_tensor-1.yaml")

        # Manual copy of test_tensor-1.yaml

        #         0    1    2    3
        #
        t = [ 100, 101, 0, 102 ]

        fiber = Fiber( [0, 1, 3], [100, 101, 102])
        tensor = Tensor.fromUncompressed(["M"], t)

        self.assertEqual(tensor.getRoot(), fiber)


    def test_fromUncompressed_2D(self):
        """Test construction of a tensor from nested lists"""

        tensor_ref = Tensor("./data/test_tensor-1.yaml")

        # Manual copy of test_tensor-1.yaml

        #         0    1    2    3
        #
        t = [ [   0,   0,   0,   0 ],  # 0
              [ 100, 101, 102,   0 ],  # 1
              [   0, 201,   0, 203 ],  # 2
              [   0,   0,   0,   0 ],  # 3
              [ 400,   0, 402,   0 ],  # 4
              [   0,   0,   0,   0 ],  # 5
              [   0, 601,   0, 603 ] ] # 6

        tensor = Tensor.fromUncompressed(["M", "K"], t)

        self.assertEqual(tensor, tensor_ref)

    def test_fromFiber(self):
        """Test construction of a tensor from a fiber"""

        tensor_ref = Tensor("./data/test_tensor-1.yaml")

        root = tensor_ref.getRoot()

        tensor = Tensor.fromFiber(["M", "K"], root)

        self.assertEqual(tensor, tensor_ref)

    def test_setRoot(self):
        """Test adding a new root"""

        tensor_ref = Tensor("./data/test_tensor-1.yaml")

        root = tensor_ref.getRoot()

        tensor = Tensor(rank_ids=["M", "K"])
        tensor.setRoot(root)

        self.assertEqual(tensor, tensor_ref)

    def test_values(self):
        """Test counting values in a tensor"""

        tensor = Tensor("./data/test_tensor-1.yaml")

        count = tensor.countValues()

        self.assertEqual(count, 9)

    def test_dump(self):
        """Test dumping a tensor"""

        tensor = Tensor("./data/test_tensor-1.yaml")
        tensor.dump("/tmp/test_tensor-1.yaml")

        tensor_tmp = Tensor("/tmp/test_tensor-1.yaml")

        self.assertTrue(tensor == tensor_tmp)


if __name__ == '__main__':
    unittest.main()

