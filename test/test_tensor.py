import unittest

from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.rank import Rank
from fibertree.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_constructor_empty(self):
        """Test construction of empty tensor"""

        Tensor(rank_ids=["M", "K"])


    def test_constructor_rank_0D(self):
        """Test construction of 0-D tensor"""

        t = Tensor(rank_ids=[])
        p = t.getRoot()
        p += 1

        self.assertEqual(p, 1)


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


    def test_fromYAMLfile_0D(self):
        """Test construction of 0-D tensor from a YAML file"""

        tensor_ref  = Tensor(rank_ids=[])
        root = tensor_ref.getRoot()
        root += 2

        tensor = Tensor.fromYAMLfile("./data/tensor_0d.yaml")

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

    def test_fromUncompressed_20(self):
        """Test construction of a tensor a scalar"""

        t_ref = Tensor(rank_ids=[])
        p = t_ref.getRoot()
        p += 2

        t0 = Tensor.fromUncompressed([], 2)

        self.assertEqual(t0, t_ref)

        t1 = Tensor.fromUncompressed(rank_ids=[], root=2)

        self.assertEqual(t1, t_ref)

        t2 = Tensor.fromUncompressed(root=2)

        self.assertEqual(t2, t_ref)


    def test_fromFiber(self):
        """Test construction of a tensor from a fiber"""

        tensor_ref = Tensor("./data/test_tensor-1.yaml")

        root = tensor_ref.getRoot()

        tensor = Tensor.fromFiber(["M", "K"], root)

        self.assertEqual(tensor, tensor_ref)


    def test_print_0D(self):
        """Test printing a 0-D tensor"""

        a = Tensor(rank_ids=[])
        p = a.getRoot()
        p += 2

        a_s_ref = "T()/[<2>]"

        a_s = f"{a}"

        self.assertEqual(a_s, a_s_ref)

        a_r_ref = "T()/[2]"

        a_r = f"{a!r}"

        self.assertEqual(a_r, a_r_ref)


    def test_print_2D(self):
        """Test printing a 2-D tensor"""

        a = Tensor.fromYAMLfile("./data/matrix-a.yaml")

        a_s_ref = "T(M,K)/[\n" + \
                  "  Rank: M F(M)/[( 0 -> F(K)/[(0 -> <1>) \n" + \
                  "                             (2 -> <3>) ])\n" + \
                  "                ( 1 -> F(K)/[(0 -> <1>) \n" + \
                  "                             (3 -> <4>) ])\n" + \
                  "                ( 3 -> F(K)/[(2 -> <3>) \n" + \
                  "                             (3 -> <4>) ])\n" + \
                  "  Rank: K F(K)/[(0 -> <1>) \n" + \
                  "                (2 -> <3>) ],\n" + \
                  "          F(K)/[(0 -> <1>) \n" + \
                  "                (3 -> <4>) ],\n" + \
                  "          F(K)/[(2 -> <3>) \n" + \
                  "                (3 -> <4>) ]\n" + \
                  "]"

        a_s = f"{a}"

        self.assertEqual(a_s, a_s_ref)

        a_r_ref = "T(M,K)/[\n" + \
                  "  R(M)/[Fiber([0, 1, 3], [Fiber([0, 2], [1, 3], owner=K), Fiber([0, 3], [1, 4], owner=K), Fiber([2, 3], [3, 4], owner=K)], owner=M)]\n" + \
                  "  R(K)/[Fiber([0, 2], [1, 3], owner=K), Fiber([0, 3], [1, 4], owner=K), Fiber([2, 3], [3, 4], owner=K)]\n" + \
                  "]"
        a_r = f"{a!r}"

        self.assertEqual(a_r, a_r_ref)


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

