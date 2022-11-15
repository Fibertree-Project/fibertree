import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Metrics
from fibertree import Rank
from fibertree import Tensor


class TestTensor(unittest.TestCase):

    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

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

        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        # Filename no longer becomes tensor name
        # self.assertEqual(t.getName(), "test_tensor-1")

        self.assertEqual(t.getRankIds(),[ "M", "K" ])
        self.assertEqual(t.getShape(), [7, 4])


    def test_equal(self):
        """Test equality comparison"""

        tensor1 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        tensor2 = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        self.assertTrue(tensor1 == tensor2)

    def test_fromYAML(self):
        """Test construction from a YAML file"""

        tensor_ref  = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

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

        tensor_ref = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        # Manual copy of test_tensor-1.yaml

        #         0    1    2    3
        #
        t = [ 100, 101, 0, 102 ]

        fiber = Fiber( [0, 1, 3], [100, 101, 102])
        tensor = Tensor.fromUncompressed(["M"], t)

        self.assertEqual(tensor.getRoot(), fiber)


    def test_fromUncompressed_2D(self):
        """Test construction of a tensor from nested lists"""

        tensor_ref = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

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

    def test_fromUncompressed_2D_wo_ids(self):
        """Test construction of a tensor from nested lists without ids"""

        tensor_in = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        root = tensor_in.getRoot()
        tensor_ref = Tensor.fromFiber(["R1", "R0"], root)

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

        tensor = Tensor.fromUncompressed(["R1", "R0"], t)

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

        tensor_ref = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        root = tensor_ref.getRoot()

        tensor = Tensor.fromFiber(["M", "K"], root)

        self.assertEqual(tensor, tensor_ref)

    def test_fromFiber_wo_ids(self):
        """Test construction of a tensor from a fiber without rank ids"""

        tensor_in = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        root = tensor_in.getRoot()
        tensor_ref = Tensor.fromFiber(["R1", "R0"], root)

        tensor = Tensor.fromFiber(fiber=root)

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


    def test_fromRandom_wo_ids(self):
        """Test construction of a random tensor without rankids"""

        rank_ids = ["R1", "R0"]
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

        tensor = Tensor.fromRandom(None, shape, [0.5, 0.5], 10, seed=3)

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

        a_r_ref = "T()/[Payload(2)]"

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

        tensor_ref = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        root = tensor_ref.getRoot()

        tensor = Tensor(rank_ids=["M", "K"])
        tensor.setRoot(root)

        self.assertEqual(tensor, tensor_ref)


    def test_getPayload_0d(self):
        """Test getPayload of a 0-D tensor"""

        p_ref = 10

        t = Tensor(rank_ids=[])
        r = t.getRoot()
        r <<= p_ref

        p = t.getPayload()
        self.assertEqual(p_ref, p)

        p = t.getPayload(0)
        self.assertEqual(p_ref, p)

        p = t.getPayload(1)
        self.assertEqual(p_ref, p)



    def test_getPayload_2d(self):
        """Test getPayload of a 2-D tensor"""

        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        with self.subTest(test="Existing element"):
            p23_ref = 203
            p23 = t.getPayload(2, 3)

            self.assertEqual(p23_ref, p23)

            # Make sure change is seen
            p23_new_ref = 310
            p23 <<= p23_new_ref

            p23_new = t.getPayload(2, 3)

            self.assertEqual(p23_new_ref, p23_new)


        with self.subTest(test="Non-existing element"):
            p31_ref = 0
            p31 = t.getPayload(3, 1)

            self.assertEqual(p31_ref, p31)

            # Make sure change is NOT seen
            p31_new_ref = 100

            p31 <<= p31_new_ref
            p31_new = t.getPayload(3, 1)

            self.assertEqual(0, p31_new)


        with self.subTest(test="Element of non-existing fiber"):
            p51_ref = 0
            p51 = t.getPayload(5, 1)

            self.assertEqual(p51_ref, p51)

            # Make sure change is NOT seen
            p51_new_ref = 100

            p51 <<= p51_new_ref
            p51_new = t.getPayload(5, 1)

            self.assertEqual(0, p51_new)


        with self.subTest(test="Existing fiber"):
            p4_ref = Fiber([0, 2], [400, 402])
            p4 = t.getPayload(4)

            self.assertEqual(p4_ref, p4)


    def test_getPayloadRef_0d(self):
        """Test getPayloadRef of a 0-D tensor"""

        p_ref = 10

        t = Tensor(rank_ids=[])
        r = t.getRoot()
        r <<= p_ref

        p = t.getPayloadRef()
        self.assertEqual(p_ref, p)

        p = t.getPayloadRef(0)
        self.assertEqual(p_ref, p)

        p = t.getPayloadRef(1)
        self.assertEqual(p_ref, p)



    def test_getPayloadRef_2d(self):
        """Test getPayloadRef of a 2-D tensor"""

        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        with self.subTest(test="Existing element"):
            p23_ref = 203
            p23 = t.getPayloadRef(2, 3)

            self.assertEqual(p23_ref, p23)

            # Make sure change is seen
            p23_new_ref = 310
            p23 <<= p23_new_ref

            p23_new = t.getPayload(2, 3)

            self.assertEqual(p23_new_ref, p23_new)


        with self.subTest(test="Non-existing element"):
            p31_ref = 0
            p31 = t.getPayloadRef(3, 1)

            self.assertEqual(p31_ref, p31)

            # Make sure change is seen
            p31_new_ref = 100

            p31 <<= p31_new_ref
            p31_new = t.getPayload(3, 1)

            self.assertEqual(p31_new_ref, p31_new)


        with self.subTest(test="Element of non-existing fiber"):
            p51_ref = 0
            p51 = t.getPayloadRef(5, 1)

            self.assertEqual(p51_ref, p51)

            # Make sure change is NOT seen
            p51_new_ref = 100

            p51 <<= p51_new_ref
            p51_new = t.getPayload(5, 1)

            self.assertEqual(p51_new_ref, p51_new)


        with self.subTest(test="Existing fiber"):
            p4_ref = Fiber([0, 2], [400, 402])
            p4 = t.getPayloadRef(4)

            self.assertEqual(p4_ref, p4)


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

        tensor = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        count = tensor.countValues()

        self.assertEqual(count, 9)

    def test_dump(self):
        """Test dumping a tensor"""

        tensor = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        tensor.dump("/tmp/test_tensor-1.yaml")

        tensor_tmp = Tensor.fromYAMLfile("/tmp/test_tensor-1.yaml")

        self.assertTrue(tensor == tensor_tmp)

    def test_init_mutable(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        self.assertFalse(t.isMutable())

        t2 = Tensor(rank_ids=["X", "Y", "Z"])
        self.assertTrue(t2.isMutable())

    def test_mutable(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t.setMutable(True)
        self.assertTrue(t.isMutable())

    def test_mutable_after_split(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.splitUniform(10)
        self.assertFalse(t2.isMutable())

        t3 = Tensor(rank_ids=["X", "Y", "Z"])
        t4 = t3.splitUniform(10)
        self.assertTrue(t4.isMutable())

    def test_mutable_after_swap(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.swapRanks()
        self.assertFalse(t2.isMutable())

        t3 = Tensor(rank_ids=["X", "Y", "Z"])
        t4 = t3.swapRanks()
        self.assertTrue(t4.isMutable())

    def test_mutable_after_flatten(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.flattenRanks()
        self.assertFalse(t2.isMutable())

        t3 = Tensor(rank_ids=["X", "Y", "Z"])
        t4 = t3.flattenRanks()
        self.assertTrue(t4.isMutable())

    def test_mutable_after_unflatten(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.flattenRanks()
        t3 = t2.unflattenRanks()
        self.assertFalse(t3.isMutable())

        t4 = Tensor(rank_ids=["X", "Y", "Z"])
        t5 = t4.flattenRanks()
        t6 = t5.unflattenRanks()
        self.assertTrue(t6.isMutable())

    def test_get_set_format(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        self.assertEqual(t.getFormat("M"), "C")
        self.assertEqual(t.getFormat("K"), "C")

        t.setFormat("K", "U")

        self.assertEqual(t.getFormat("M"), "C")
        self.assertEqual(t.getFormat("K"), "U")

        self.assertRaises(ValueError, lambda: t.getFormat("N"))
        self.assertRaises(ValueError, lambda: t.setFormat("N", "C"))
        self.assertRaises(AssertionError, lambda: t.setFormat("M", "G"))

    def test_format_after_split(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t.setFormat("K", "U")
        t2 = t.splitUniform(5, depth=1)
        t3 = t2.splitUniform(6, depth=0)

        self.assertEqual(t3.getFormat("M.1"), "C")
        self.assertEqual(t3.getFormat("M.0"), "C")
        self.assertEqual(t3.getFormat("K.1"), "U")
        self.assertEqual(t3.getFormat("K.0"), "U")

    def test_format_after_swap(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t.setFormat("K", "U")
        t2 = t.swapRanks()

        self.assertEqual(t2.getFormat("M"), "C")
        self.assertEqual(t2.getFormat("K"), "U")

    def test_format_after_flatten(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.splitUniform(2, depth=1)
        t2.setFormat("M", "U")
        t2.setFormat("K.0", "U")

        t3 = t2.flattenRanks()
        self.assertEqual(t3.getFormat(["M", "K.1"]), "C")
        self.assertEqual(t3.getFormat("K.0"), "U")

    def test_format_after_unflatten(self):
        t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        t2 = t.splitUniform(2, depth=1)
        t2.setFormat("M", "U")
        t2.setFormat("K.0", "U")

        t3 = t2.flattenRanks()
        t4 = t3.unflattenRanks()
        self.assertEqual(t4.getFormat("M"), "C")
        self.assertEqual(t4.getFormat("K.1"), "C")
        self.assertEqual(t4.getFormat("K.0"), "U")

    def test_flattenRanks_corr_shape(self):
        """Test that flattenRanks sets the correct shape"""
        t = Tensor(rank_ids=["J", "K", "M", "N", "O"], shape=[2, 3, 4, 5, 6])

        tuple_ = t.flattenRanks(depth=1, levels=2, coord_style="tuple")
        self.assertEqual(tuple_.getShape(), [2, (3, 4, 5), 6])

        pair = t.flattenRanks(depth=1, levels=2, coord_style="pair")
        self.assertEqual(pair.getShape(), [2, (3, (4, 5)), 6])

        absolute = t.flattenRanks(depth=1, levels=2, coord_style="absolute")
        self.assertEqual(absolute.getShape(), [2, 5, 6])

        # Note, this does not really make sense because it is not a good use
        # of relative; Imagine [10] relative partitioned with splitUniform(2);
        # This should have shape [10, 2]; flattening should remake [10]
        relative = t.flattenRanks(depth=1, levels=2, coord_style="relative")
        self.assertEqual(relative.getShape(), [2, 3, 6])

        linear = t.flattenRanks(depth=1, levels=2, coord_style="linear")
        self.assertEqual(linear.getShape(), [2, 60, 6])


if __name__ == '__main__':
    unittest.main()
