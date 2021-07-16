import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Rank
from fibertree import Tensor


class TestTensorTransform(unittest.TestCase):

    def test_truediv(self):
        """ Test /, the __truediv__ operator """
        a = Tensor.fromYAMLfile("./data/tensor_transform-b.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-b-truediv.yaml")

        a_out = a / 4

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N"])
        self.assertEqual(a_out.getShape(), [16, 20, 10])

    def test_splitUniform_0(self):
        """ Test splitUniform - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUniform_0.yaml")

        tests = { "by-depth": {"depth": 0},
                  "by-name":  {"rankid": "M"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUniform(25, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
                self.assertEqual(a_out.getShape(), [26, 41, 42, 10])



    def test_splitUniform_1(self):
        """ Test splitUniform - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUniform_1.yaml")

        tests = { "by-depth": {"depth": 1},
                  "by-name":  {"rankid": "N"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUniform(15, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
                self.assertEqual(a_out.getShape(), [41, 31, 42, 10])


    def test_splitUniform_2(self):
        """ Test splitUniform - depth=2 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUniform_2.yaml")
        tests = { "by-depth": {"depth": 2},
                  "by-name":  {"rankid": "K"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUniform(4, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
                self.assertEqual(a_out.getShape(), [41, 42, 9, 10])


    def test_splitNonUniform_0(self):
        """ Test splitNonUniform - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitNonUniform_0.yaml")

        tests = { "by-depth": {"depth": 0},
                  "by-name":  {"rankid": "M"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitNonUniform([0, 15, 35], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
                self.assertEqual(a_out.getShape(), [36, 41, 42, 10])


    def test_splitNonUniform_1(self):
        """ Test splitNonUniform - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitNonUniform_1.yaml")

        tests = { "by-depth": {"depth": 1},
                  "by-name":  {"rankid": "N"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitNonUniform([0, 15, 25], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
                self.assertEqual(a_out.getShape(), [41, 26, 42, 10])


    def test_splitNonUniform_2(self):
        """ Test splitNonUniform - depth=2 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitNonUniform_2.yaml")

        tests = { "by-depth": {"depth": 2},
                  "by-name":  {"rankid": "K"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitNonUniform([0, 4, 19], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
                self.assertEqual(a_out.getShape(), [41, 42, 5, 10])

    def test_floordiv(self):
        """ Test /, the __floordiv__ operator """
        a = Tensor.fromYAMLfile("./data/tensor_transform-b.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-b-floordiv.yaml")

        a_out = a // 4

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N"])
        self.assertEqual(a_out.getShape(), [17, 20, 10])

    def test_splitEqual_0(self):
        """ Test splitEqual - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitEqual_0.yaml")

        tests = { "by-depth": {"depth": 0},
                  "by-name":  {"rankid": "M"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitEqual(2, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
                self.assertEqual(a_out.getShape(), [41, 41, 42, 10])


    def test_splitEqual_1(self):
        """ Test splitEqual - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitEqual_1.yaml")

        tests = { "by-depth": {"depth": 1},
                  "by-name":  {"rankid": "N"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitEqual(2, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
                self.assertEqual(a_out.getShape(), [41, 34, 42, 10])


    def test_splitEqual_2(self):
        """ Test splitEqual - depth=2 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitEqual_2.yaml")

        tests = { "by-depth": {"depth": 2},
                  "by-name":  {"rankid": "K"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitEqual(2, **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
                self.assertEqual(a_out.getShape(), [41, 42, 10, 10])



    def test_splitUnEqual_0(self):
        """ Test splitUnEqual - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUnEqual_0.yaml")

        tests = { "by-depth": {"depth": 0},
                  "by-name":  {"rankid": "M"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUnEqual([2, 1], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M.1", "M.0", "N", "K"])
                self.assertEqual(a_out.getShape(), [41, 41, 42, 10])



    def test_splitUnEqual_1(self):
        """ Test splitUnEqual - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUnEqual_1.yaml")

        tests = { "by-depth": {"depth": 1},
                  "by-name":  {"rankid": "N"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUnEqual([2, 1], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N.1", "N.0", "K"])
                self.assertEqual(a_out.getShape(), [41, 42, 42, 10])


    def test_splitUnEqual_2(self):
        """ Test splitUnEqual - depth=2 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-splitUnEqual_2.yaml")

        tests = { "by-depth": {"depth": 2},
                  "by-name":  {"rankid": "K"}}

        for test, kwargs in tests.items():
            with self.subTest(test=test):
                a_out = a.splitUnEqual([2, 1], **kwargs)

                self.assertEqual(a_out, a_verify)
                self.assertEqual(a_out.getRankIds(), ["M", "N", "K.1", "K.0"])
                self.assertEqual(a_out.getShape(), [41, 42, 9, 10])


    def test_swapRanks_0(self):
        """ Test swapRanks - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-swapRanks_0.yaml")

        a_out = a.swapRanks(depth=0)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["N", "M", "K"])
        self.assertEqual(a_out.getShape(), [42, 41, 10])


    def test_swapRanks_1(self):
        """ Test swapRanks - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_verify = Tensor.fromYAMLfile("./data/tensor_transform-a-swapRanks_1.yaml")

        a_out = a.swapRanks(depth=1)

        self.assertEqual(a_out, a_verify)
        self.assertEqual(a_out.getRankIds(), ["M", "K", "N"])
        self.assertEqual(a_out.getShape(), [41, 10, 42])


    def test_swizzleRanks(self):
        """ Test swizzleRanks """

        a_MK = Tensor.fromUncompressed(["M", "K"],
                               [[0, 0, 4, 0, 0, 5],
                                [3, 2, 0, 3, 0, 2],
                                [0, 2, 0, 0, 1, 2],
                                [0, 0, 0, 0, 0, 0],
                                [2, 5, 0, 0, 0, 5],
                                [4, 1, 0, 0, 0, 0],
                                [5, 0, 0, 1, 0, 0],
                                [4, 0, 0, 5, 1, 3]])

        a_KM = a_MK.swapRanks()

        M = 8
        M1 = 2
        M0 = (M+1)//M1

        K = 6
        K1 = 2
        K0 = (K+1)//K1

        a_MMKK = a_MK.splitUniform(M0).splitUniform(K0, depth=2)
        a_MKMK = a_MMKK.swapRanks(depth=1)
        a_KMMK = a_KM.splitUniform(K0).swapRanks(depth=1).splitUniform(M0, depth=1)

        a_KM_2 = a_MK.swizzleRanks(["K", "M"])
        self.assertEqual(a_KM_2, a_KM)

        a_MK_2 = a_KM_2.swizzleRanks(["M", "K"])
        self.assertEqual(a_MK_2, a_MK)

        a_MKMK_2 = a_MMKK.swizzleRanks(["M.1","K.1", "M.0", "K.0"])
        self.assertEqual(a_MKMK_2, a_MKMK)

        a_MMKK_2 = a_MKMK.swizzleRanks(["M.1", "M.0", "K.1", "K.0"])
        self.assertEqual(a_MMKK_2, a_MMKK)

    def test_swizzleRanks_empty(self):
        """ Test swizzleRanks() on an empty tensor """
        Z_MNOP = Tensor(rank_ids=["M", "N", "O", "P"])
        Z_PNMO = Z_MNOP.swizzleRanks(rank_ids=["P", "N", "M", "O"])

        self.assertEqual(Z_MNOP.getRankIds(), ["M", "N", "O", "P"])
        self.assertEqual(Z_PNMO.getRankIds(), ["P", "N", "M", "O"])


    def test_flattenRanks_0(self):
        """ Test flattenRanks - depth=0 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
        a_out = a.flattenRanks(depth=0)
        a_again = a_out.unflattenRanks(depth=0)

        self.assertEqual(a_again, a)
        self.assertEqual(a_out.getRankIds(), [["M", "N"], "K"])

        # TBD: Semantics for non-integer coordinates
#       self.assertEqual(a_out.getShape(), [7, 10])


    def test_flattenRanks_1(self):
        """ Test flattenRanks - depth=1 """

        a = Tensor.fromYAMLfile("./data/tensor_transform-a.yaml")
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

    def test_flattenRanks_l3_sa(self):
        """Test flattenRanks - levels=3, coord_style=absolute"""
        t0 = Tensor.fromUncompressed(rank_ids=["A"], root=list(range(16)))
        s1 = t0.splitUniform(8, depth=0)
        s2 = s1.splitUniform(4, depth=1)
        s3 = s2.splitUniform(2, depth=2)

        f4 = s3.flattenRanks(levels=3, coord_style="absolute")
        f4.setRankIds(["A"])

        self.assertEqual(f4, t0)

    def test_unflattenRanks_empty(self):
        t = Tensor(rank_ids=["X", "Y", "Z"])
        t2 = t.flattenRanks()
        t3 = t2.unflattenRanks()
        t3.setRankIds(["X", "Y", "Z"])

        self.assertEqual(t, t3)

if __name__ == '__main__':
    unittest.main()

