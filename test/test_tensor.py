import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Rank
from fibertree import Tensor


class TestTensor(unittest.TestCase):

    def test_constructor_empty(self):
        """Test construction of empty tensor"""

        ranks = ["M", "K"]

        t = Tensor(rank_ids=ranks)
        self.assertEqual(t.getRankIds(), ranks)
        self.assertEqual(t.getRoot().getRankIds(), ranks)

    def test_constructor_shape(self):
        """Test construction of shape of tensor"""

        ranks = ["M", "K"]
        shape = [4, 8]

        t = Tensor(rank_ids=ranks, shape=shape)

        self.assertEqual(t.getRankIds(), ranks)
        self.assertEqual(t.getRoot().getRankIds(), ranks)

        self.assertEqual(t.getShape(), shape)
        self.assertEqual(t.getRoot().getShape(), shape)

        
    def test_constructor_shape(self):
        """Test construction of shape of tensor"""

        ranks = ["M", "K"]
        name = "ME"

        t1 = Tensor(rank_ids=ranks, name=name)

        self.assertEqual(t1.getName(), name)

        t2 = Tensor(rank_ids=ranks)
        t2.setName(name)

        self.assertEqual(t2.getName(), name)


    def test_constructor_rank_0D(self):
        """Test construction of 0-D tensor"""

        t = Tensor(rank_ids=[])
        p = t.getRoot()
        p += 1

        self.assertEqual(p, 1)


    def test_new(self):
        """Test construction of a tensor from a file"""

        t = Tensor("./data/test_tensor-1.yaml")

        self.assertEqual(t.getName(), "test_tensor-1")
        self.assertEqual(t.getRankIds(),[ "M", "K" ])
        self.assertEqual(t.getShape(), [7, 4])


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


    def test_fomYAMLfile_3D(self):
        """Test construction of 0-D tensor from a YAML file"""

        t = Tensor.fromYAMLfile("./data/tensor_3d-0.yaml")

        # TBD: Check that data is good

        rankids_ref = ["M", "N", "K"]
        shape_ref = [21, 51, 11]

        self.assertEqual(t.getRankIds(), rankids_ref)
        self.assertEqual(t.getShape(), shape_ref)


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


    def test_fromRandom(self):
        """Test construction of a random tensor"""

        rank_ids = ["X", "Y"]
        shape = [10, 10]
        tensor_ref = Tensor.fromUncompressed(rank_ids,
                                             [[0, 10, 10, 1, 0, 9, 8, 0, 0, 3],
                                              [9, 1, 0, 10, 1, 0, 10, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 3, 0, 3, 5, 0, 5, 7, 0, 0],
                                              [6, 0, 0, 0, 0, 0, 6, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 2, 8, 2, 3, 7, 0, 0, 10],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 4, 0, 2, 9, 4, 0, 5],
                                              [6, 3, 0, 8, 0, 10, 0, 9, 4, 0]])
        
        tensor = Tensor.fromRandom(rank_ids, shape, [0.5, 0.5], 10, seed=3)

        self.assertEqual(tensor, tensor_ref)
        self.assertEqual(tensor.getRankIds(), rank_ids)


    def test_print_0D(self):
        """Test printing a 0-D tensor"""

        a = Tensor(rank_ids=[])
        p = a.getRoot()
        p += 2

        a_s_ref = "<2>"

        a_s = f"{a}"

        self.assertEqual(a_s, a_s_ref)

        a_r_ref = "T()/[2]"

        a_r = f"{a!r}"

        self.assertEqual(a_r, a_r_ref)


    def test_print_2D(self):
        """Test printing a 2-D tensor"""

        a = Tensor.fromYAMLfile("./data/matrix-a.yaml")

#
# Old style print
#
#        a_s_ref = "T(M,K)/[\n" + \
#                  "  Rank: M F(M)/[( 0 -> F(K)/[(0 -> <1>) \n" + \
#                  "                             (2 -> <3>) ])\n" + \
#                  "                ( 1 -> F(K)/[(0 -> <1>) \n" + \
#                  "                             (3 -> <4>) ])\n" + \
#                  "                ( 3 -> F(K)/[(2 -> <3>) \n" + \
#                  "                             (3 -> <4>) ])\n" + \
#                  "  Rank: K F(K)/[(0 -> <1>) \n" + \
#                  "                (2 -> <3>) ],\n" + \
#                  "          F(K)/[(0 -> <1>) \n" + \
#                  "                (3 -> <4>) ],\n" + \
#                  "          F(K)/[(2 -> <3>) \n" + \
#                  "                (3 -> <4>) ]\n" + \
#                  "]"

        a_s_ref = "F(M)/[( 0 -> F(K)/[(0 -> <1>) (2 -> <3>) ])( 1 -> F(K)/[(0 -> <1>) (3 -> <4>) ])......"

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


    def test_default(self):
        """Test of default default"""

        t = Tensor(rank_ids=["X", "Y", "Z"])

        e = Fiber([], [])

        self.assertEqual(t.getDefault(), 0)

        t_root = t.getRoot()

        x = t_root.getPayload(1)
        self.assertEqual(x, e)
        self.assertEqual(x.getDefault(), Fiber)

        y = t_root.getPayload(1, 2)
        self.assertEqual(y, e)
        self.assertEqual(y.getDefault(), 0)

        z = t_root.getPayload(1, 2, 3)
        self.assertEqual(z, 0)


    def test_default_nonzero(self):
        """Test set/get of nonzero default"""

        t = Tensor(rank_ids=["X", "Y", "Z"])

        v = 10
        e = Fiber([], [])

        t.setDefault(v)
        self.assertEqual(t.getDefault(), v)

        t_root = t.getRoot()

        x = t_root.getPayload(1)
        self.assertEqual(x, e)

        y = t_root.getPayload(1, 2)
        self.assertEqual(y, e)

        z = t_root.getPayload(1, 2, 3)
        self.assertEqual(z, v)


    def test_default_nonscalar(self):
        """Test set/get of nonzero default"""

        t = Tensor(rank_ids=["X", "Y", "Z"])

        v = (10, 10)
        e = Fiber([], [])

        t.setDefault(v)
        self.assertEqual(t.getDefault(), v)

        t_root = t.getRoot()

        x = t_root.getPayload(1)
        self.assertEqual(x, e)

        y = t_root.getPayload(1, 2)
        self.assertEqual(y, e)

        z = t_root.getPayload(1, 2, 3)
        self.assertEqual(z, v)


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

