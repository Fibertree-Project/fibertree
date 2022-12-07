""" Tests of fiber operators """

import os
import unittest

from fibertree import Fiber
from fibertree import Metrics
from fibertree import Payload
from fibertree import Tensor

class TestFiberOperators(unittest.TestCase):
    """ Tests of fiber operators """

    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

        # Make sure we have a tmp directory to write to
        if not os.path.exists("tmp"):
            os.makedirs("tmp")


    def test_add_int(self):
        """Test __add__ integers"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber.fromUncompressed([3, 4, 5, 2, 2, 8])

        with self.subTest("f_in + 2"):
            f_out = f_in + 2
            self.assertEqual(f_ref, f_out)

        with self.subTest("2 + f_in"):
            f_out = 2 + f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in += 2"):
            # f_in gets clobbered!
            f_in += 2
            self.assertEqual(f_ref, f_in)


    def test_add_payload(self):
        """Test __add__ payload"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber.fromUncompressed([3, 4, 5, 2, 2, 8])
        two = Payload(2)

        with self.subTest("f_in + 2"):
            f_out = f_in + two
            self.assertEqual(f_ref, f_out)

        with self.subTest("2 + f_in"):
            f_out = two + f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in += 2"):
            # f_in gets clobbered!
            f_in += two
            self.assertEqual(f_ref, f_in)


    def test_add_fiber(self):
        """Test __add__ fiber"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        g_in = Fiber([6, 8], [20, 22])
        fg_ref = Fiber([0, 1, 2, 5, 6, 8], [1, 2, 3, 6, 20, 22])

        with self.subTest("f+g"):
            fg_out = f_in + g_in
            self.assertEqual(fg_ref, fg_out)

        with self.subTest("f+=g"):
            # f_in gets clobbered!
            f_in += g_in
            self.assertEqual(fg_ref, f_in)

    def test_add_eager_only(self):
        """Test __add__ for eager only"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_in._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f_in + 1

    def test_and_fiber(self):
        """Test __and__ fiber"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 4, 5, 0, 7])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([9, 2, 3, 8, 0, 6, 0])
        b_k.getRankAttrs().setId("K")

        cc = [0, 3]
        cp = [(1, 9), (4, 8)]
        self.assertEqual(len(cc), len(a_k & b_k))
        for i, (c, p) in enumerate(a_k & b_k):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

        # Check the fiber attributes
        id_ = (a_k & b_k).getRankAttrs().getId()
        self.assertEqual(id_, "K")
        self.assertEqual((a_k & b_k).getActive(), (0, 7))

    def test_and_tuple_coords(self):
        """Test __and__ with tuple coordinates"""
        a_km = Fiber([(0, 0), (0, 2), (3, 5), (4, 8)], [1, 2, 3, 4])
        b_km = Fiber([(0, 0), (0, 1), (4, 8), (4, 9)], [1, 2, 3, 4])

        cc = [(0, 0), (4, 8)]
        cp = [(1, 1), (4, 3)]
        self.assertEqual(len(cc), len(a_km & b_km))

        for i, (c, p) in enumerate(a_km & b_km):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

    def test_and_tuple_a_longer(self):
        """Test __and__ with tuple coords, but a's tuple is longer"""
        a_km = Fiber([(0, 0), (0, 2), (3, 5), (4, 8)], [1, 2, 3, 4])
        b_k = Fiber([(0,), (1,), (2,), (3,)], [1, 2, 3, 4], active_range=((0,), (4,)))

        cc = [(0, 0), (0, 2), (3, 5)]
        cp = [(1, 1), (2, 1), (3, 4)]
        self.assertEqual(len(cc), len(a_km & b_k))

        for i, (c, p) in enumerate(a_km & b_k):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

    def test_and_a_tuple(self):
        """Test __and__ with a having tuple coordinates"""
        a_km = Fiber([(0, 0), (0, 2), (3, 5), (4, 8)], [1, 2, 3, 4])
        b_k = Fiber([0, 1, 2, 3], [1, 2, 3, 4])

        cc = [(0, 0), (0, 2), (3, 5)]
        cp = [(1, 1), (2, 1), (3, 4)]
        self.assertEqual(len(cc), len(a_km & b_k))

        for i, (c, p) in enumerate(a_km & b_k):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

    def test_and_tuple_b_longer(self):
        """Test __and__ with tuple coords, but b's tuple is longer"""
        a_k = Fiber([(0,), (1,), (2,), (3,)], [1, 2, 3, 4], active_range=((0,), (4,)))
        b_km = Fiber([(0, 0), (0, 2), (3, 5), (4, 8)], [1, 2, 3, 4])

        cc = [(0, 0), (0, 2), (3, 5)]
        cp = [(1, 1), (1, 2), (4, 3)]
        self.assertEqual(len(cc), len(a_k & b_km))

        for i, (c, p) in enumerate(a_k & b_km):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

    def test_and_tuple_b_tuple(self):
        """Test __and__ with b having tuple coordinates"""
        a_k = Fiber([0, 1, 2, 3], [1, 2, 3, 4])
        b_km = Fiber([(0, 0), (0, 2), (3, 5), (4, 8)], [1, 2, 3, 4])

        cc = [(0, 0), (0, 2), (3, 5)]
        cp = [(1, 1), (1, 2), (4, 3)]
        self.assertEqual(len(cc), len(a_k & b_km))

        for i, (c, p) in enumerate(a_k & b_km):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

    def test_and_metrics_fiber(self):
        """Test metrics collected during Fiber.__and__ on unowned fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 4, 5, 6, 7])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([1, 2, 0, 0, 0, 6, 0])
        b_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_and_metrics_fiber")
        Metrics.trace("K", type_="intersect_0")
        Metrics.trace("K", type_="intersect_1")
        for _ in a_k & b_k:
            pass
        Metrics.endCollect()

        corr0 = [
            "K_pos,K,fiber_pos\n",
            "0,0,0\n",
            "2,3,1\n",
            "3,4,2\n",
            "4,5,3\n",
            "5,6,4\n"
        ]

        corr1 = [
            "K_pos,K,fiber_pos\n",
            "0,0,0\n",
            "1,1,1\n",
            "4,5,2\n"
        ]

        with open("tmp/test_and_metrics_fiber-K-intersect_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr0)

        with open("tmp/test_and_metrics_fiber-K-intersect_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr1)

    def test_and_metrics_tensor(self):
        """Test metrics collected during Fiber.__and__ on unowned fibers"""
        A_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 0, 4, 5, 6, 7])
        B_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 2, 0, 0, 0, 6, 0])

        Metrics.beginCollect("tmp/test_and_metrics_tensor")
        Metrics.trace("K", type_="intersect_0")
        Metrics.trace("K", type_="intersect_1")
        for _ in A_K.getRoot() & B_K.getRoot():
            pass
        Metrics.endCollect()

        corr0 = [
            "K_pos,K,fiber_pos\n",
            "0,0,0\n",
            "2,3,1\n",
            "3,4,2\n",
            "4,5,3\n",
            "5,6,4\n"
        ]

        corr1 = [
            "K_pos,K,fiber_pos\n",
            "0,0,0\n",
            "1,1,1\n",
            "4,5,2\n"
        ]

        with open("tmp/test_and_metrics_tensor-K-intersect_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr0)

        with open("tmp/test_and_metrics_tensor-K-intersect_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr1)

    def test_and_metrics_many_fibers(self):
        """Test metrics collection when there are more than two fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 0, 6])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([1, 0, 3, 0, 5, 6])
        b_k.getRankAttrs().setId("K")
        c_k = Fiber.fromUncompressed([0, 0, 0, 4, 0, 6])
        c_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_and_metrics_many_fibers")
        Metrics.trace("K", "intersect_0_1")
        Metrics.trace("K", "intersect_2_3")
        for _ in (a_k & b_k) & c_k:
            pass
        Metrics.endCollect()

        corr01 = [
            "K_pos,K,0_match,1_match\n",
            "0,0,True,True\n",
            "0,2,True,True\n",
            "0,3,True,False\n",
            "0,4,False,True\n",
            "0,5,True,True\n"
        ]

        corr23 = [
            "K_pos,K,2_match,3_match\n",
            "0,0,True,False\n",
            "0,2,True,False\n",
            "0,5,True,True\n"
        ]

    def test_and_use_stats_1D(self):
        """Test reuse statistics collected on a 1D fiber during Fiber.__and__"""
        A_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 0, 4, 5, 6])
        a_k = A_K.getRoot()

        B_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 3, 0, 5, 0])
        b_k = B_K.getRoot()

        Metrics.beginCollect("tmp/test_and_use_stats_1D")
        Metrics.trace("K")
        Metrics.registerRank("M")
        for m in range(3):
            Metrics.addUse("M", m * 2, m)
            for _ in a_k & b_k:
                pass
            Metrics.incIter("M")
        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "0,0,0,0,0\n",
            "0,3,0,4,1\n",
            "1,0,2,0,0\n",
            "1,3,2,4,1\n",
            "2,0,4,0,0\n",
            "2,3,4,4,1\n"
        ]
        with open("tmp/test_and_use_stats_1D-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_and_use_stats_2D(self):
        """Test reuse statistics collected on a 2D fiber during Fiber.__and__"""
        A_JK = Tensor.fromUncompressed(rank_ids=["J", "K"], root=[[1, 0, 3], [0, 0, 6], [0, 8, 9]])
        B_IJK = Tensor.fromUncompressed(rank_ids=["I", "J", "K"], root=[[[0, 2, 3], [0, 0, 0], [7, 8, 0]], [[1, 0, 0], [4, 5, 6], [0, 0, 0]]])

        a_j = A_JK.getRoot()
        b_i = B_IJK.getRoot()

        Metrics.beginCollect("tmp/test_and_use_stats_2D")
        Metrics.trace("J")
        for _, b_j in b_i:
            for _, (a_k, b_k) in a_j & b_j:
                for _ in a_k & b_k:
                    pass
        Metrics.endCollect()

        corr = [
            "I_pos,J_pos,I,J,fiber_pos\n",
            "0,0,0,0,0\n",
            "0,2,0,2,1\n",
            "1,0,1,0,0\n",
            "1,1,1,1,1\n"
        ]

        with open("tmp/test_and_use_stats_2D-J-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_and_with_format(self):
        """Test Fiber.__and__ with toggling the format"""
        A_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 3, 4, 0])
        a_k = A_K.getRoot()

        B_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 0, 4, 5])
        b_k = B_K.getRoot()

        # Both tensors are compressed
        A_K.setFormat("K", "C")
        B_K.setFormat("K", "C")

        inds = []
        for k, _ in a_k & b_k:
            inds.append(k)
        self.assertEqual(inds, [0, 3])

        # A is uncompressed and B is compressed
        A_K.setFormat("K", "U")
        B_K.setFormat("K", "C")

        inds = []
        for k, _ in a_k & b_k:
            inds.append(k)
        self.assertEqual(inds, [0, 3, 4])

        # A is compressed and B is uncompressed
        A_K.setFormat("K", "C")
        B_K.setFormat("K", "U")

        inds = []
        for k, _ in a_k & b_k:
            inds.append(k)
        self.assertEqual(inds, [0, 2, 3])

        # Both tensors are uncompressed
        A_K.setFormat("K", "U")
        B_K.setFormat("K", "U")

        inds = []
        for k, _ in a_k & b_k:
            inds.append(k)
        self.assertEqual(inds, [0, 1, 2, 3, 4])

    def test_lshift(self):
        """Test Fiber.__lshift__"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_m.getRankAttrs().setId("M")
        z_m = Fiber()
        z_m.getRankAttrs().setId("M")

        for _, (z_ref, a_val) in z_m << a_m:
            z_ref += a_val

        self.assertEqual(z_m, a_m)
        self.assertEqual(z_m.getActive(), (0, 5))

        id_ = (z_m << a_m).getRankAttrs().getId()
        self.assertEqual(id_, "M")
        self.assertEqual((z_m << a_m).getActive(), (0, 5))

    def test_lshift_eager_only(self):
        """Test that we can only populate eager fibers"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        z_m = Fiber.fromUncompressed([0, 2, 3, 0, 0])
        z_m._setIsLazy(True)

        with self.assertRaises(AssertionError):
            z_m << a_m

    def test_lshift_skip_empty(self):
        """Test that lshift does not insert empty fibers"""
        z_m = Tensor(rank_ids=["M", "N"]).getRoot()
        a_m = Fiber.fromUncompressed([1, 2])
        b_n = Fiber.fromUncompressed([1, 0, 3, 4, 0])

        for m, (z_n, a_val) in z_m << a_m:
            if m != 0:
                continue

            for _, (z_ref, b_val) in z_n << b_n:
                z_ref += b_val

        for z_n in z_m.payloads:
            self.assertGreater(len(z_n), 0)

        for fiber in z_m.getOwner().getFibers():
            self.assertGreater(len(fiber), 0)

        for fiber in z_m.getOwner().getNextRank().getFibers():
            self.assertGreater(len(fiber), 0)

    def test_lshift_catch_pop_wrong(self):
        """Test that lshift catches if the wrong fiber is popped off"""
        z_m = Tensor(rank_ids=["M", "N"]).getRoot()
        a_m = Fiber.fromUncompressed([1, 2])

        # Start the iterator
        iter_ = (z_m << a_m).__iter__()
        next(iter_)

        # Incorrectly change the rank
        z_m.getOwner().getNextRank().append(Fiber())

        with self.assertRaises(AssertionError):
            next(iter_)

    def test_lshift_with_start_pos(self):
        """Test that lshift works with a starting position"""
        a_m = Fiber([3, 4, 7], [1, 2, 3])
        z_m = Fiber([1, 2, 4, 8], [1, 2, 3, 4])

        for m, (z_ref, a_val) in z_m.__lshift__(a_m, start_pos=2):
            z_ref += 1

        z_corr = Fiber([1, 2, 3, 4, 7, 8], [1, 2, 1, 4, 1, 4])
        self.assertEqual(z_m, z_corr)
        self.assertEqual(z_m.getSavedPos(), 4)

    def test_lshift_bad_start_pos(self):
        """Test the failure modes for start_pos"""
        a_m = Fiber([3, 4, 7], [1, 2, 3])
        z_m = Fiber([1, 2, 4, 8], [1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            next(z_m.__lshift__(a_m, start_pos=3).__iter__())

        with self.assertRaises(AssertionError):
            next(z_m.__lshift__(a_m, start_pos=10).__iter__())

    def test_lshift_metrics_fiber(self):
        """Test metrics collection on Fiber.__lshift__ from a fiber"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_m.getRankAttrs().setId("M")
        z_m = Fiber.fromUncompressed([0, 2, 3, 0, 0])
        z_m.getRankAttrs().setId("M")

        Metrics.beginCollect("tmp/test_lshift_metrics_fiber")
        Metrics.trace("M", type_="populate_read_0")
        Metrics.trace("M", type_="populate_write_0")
        Metrics.trace("M", type_="populate_1")
        for _, (z_ref, _) in z_m << a_m:
            z_ref += 1
        Metrics.endCollect()

        corr_r0 = [
            "M_pos,M,fiber_pos\n",
            "2,1,0\n",
            "3,2,1\n",
            "7,3,6\n",
            "8,2,1\n",
            "9,1,0\n",
            "10,0,5\n"
        ]

        corr_w0 = [
            "M_pos,M,fiber_pos\n",
            "1,0,5\n",
            "4,2,1\n",
            "6,3,6\n",
            "7,3,3\n",
            "8,2,2\n",
            "9,1,1\n",
            "10,0,0\n"
        ]

        corr_r1 = [
            "M_pos,M,fiber_pos\n",
            "0,0,0\n",
            "2,2,1\n",
            "5,3,2\n"
        ]

        with open("tmp/test_lshift_metrics_fiber-M-populate_read_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r0)

        with open("tmp/test_lshift_metrics_fiber-M-populate_write_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_w0)

        with open("tmp/test_lshift_metrics_fiber-M-populate_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r1)

    def test_lshift_metrics_tensor(self):
        """Test metrics collection on Fiber.__lshift__ from a tensor"""
        A_M = Tensor.fromUncompressed(rank_ids=["M"], root=[1, 0, 3, 4, 0])
        a_m = A_M.getRoot()

        Z_M = Tensor.fromUncompressed(rank_ids=["M"], root=[0, 2, 3, 0, 0])
        z_m = Z_M.getRoot()

        Metrics.beginCollect("tmp/test_lshift_metrics_tensor")
        Metrics.trace("M", type_="populate_read_0")
        Metrics.trace("M", type_="populate_write_0")
        Metrics.trace("M", type_="populate_1")
        for _, (z_ref, _) in z_m << a_m:
            z_ref += 1
        Metrics.endCollect()

        corr_r0 = [
            "M_pos,M,fiber_pos\n",
            "2,1,0\n",
            "3,2,1\n",
            "7,3,6\n",
            "8,2,1\n",
            "9,1,0\n",
            "10,0,5\n"
        ]

        corr_w0 = [
            "M_pos,M,fiber_pos\n",
            "1,0,5\n",
            "4,2,1\n",
            "6,3,6\n",
            "7,3,3\n",
            "8,2,2\n",
            "9,1,1\n",
            "10,0,0\n"
        ]

        corr_r1 = [
            "M_pos,M,fiber_pos\n",
            "0,0,0\n",
            "2,2,1\n",
            "5,3,2\n"
        ]

        with open("tmp/test_lshift_metrics_tensor-M-populate_read_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r0)

        with open("tmp/test_lshift_metrics_tensor-M-populate_write_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_w0)

        with open("tmp/test_lshift_metrics_tensor-M-populate_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r1)

    def test_lshift_metrics_many_fibers(self):
        """Test metrics collection on Fiber.__lshift__ with more than 2 fibers"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_m.getRankAttrs().setId("M")
        z_m = Fiber.fromUncompressed([0, 2, 3, 0, 0])
        z_m.getRankAttrs().setId("M")
        y_m = Fiber()
        y_m.getRankAttrs().setId("M")

        Metrics.beginCollect("tmp/test_lshift_metrics_many_fibers")
        Metrics.trace("M", type_="populate_read_0")
        Metrics.trace("M", type_="populate_write_0")
        Metrics.trace("M", type_="populate_1")
        Metrics.trace("M", type_="populate_read_2")
        Metrics.trace("M", type_="populate_write_2")
        Metrics.trace("M", type_="populate_3")
        for _, (z_ref, (y_ref, _)) in z_m << (y_m << a_m):
            z_ref += 1
        Metrics.endCollect()

        corr_r0 = [
            "M_pos,M,fiber_pos\n",
            "2,1,0\n",
            "3,2,1\n",
            "7,3,6\n",
            "8,2,1\n",
            "9,1,0\n",
            "10,0,5\n"
        ]

        corr_w0 = [
            "M_pos,M,fiber_pos\n",
            "1,0,5\n",
            "4,2,1\n",
            "6,3,6\n",
            "7,3,3\n",
            "8,2,2\n",
            "9,1,1\n",
            "10,0,0\n"
        ]

        corr_r1 = [
            "M_pos,M,fiber_pos\n",
            "0,0,0\n",
            "2,2,1\n",
            "5,3,2\n"
        ]

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_read_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r0)

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_write_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_w0)

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r1)

        corr_r2 = [
            "M_pos,M,fiber_pos\n",
        ]

        corr_w2 = [
            "M_pos,M,fiber_pos\n",
        ]

        corr_r3 = [
            "M_pos,M,fiber_pos\n",
            "0,0,0\n",
            "2,2,1\n",
            "5,3,2\n",
        ]

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_read_2.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r2)

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_write_2.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_w2)

        with open("tmp/test_lshift_metrics_many_fibers-M-populate_3.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r3)

    def test_lshift_metrics_fiber_project(self):
        """Test metrics collection on Fiber.__lshift__ from a fiber"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_m.getRankAttrs().setId("M")
        z_n = Fiber.fromUncompressed([0, 2, 3, 0, 0])
        z_n.getRankAttrs().setId("N")

        Metrics.beginCollect("tmp/test_lshift_metrics_fiber_project")
        Metrics.trace("N", type_="populate_read_0")
        Metrics.trace("N", type_="populate_write_0")
        Metrics.trace("N", type_="populate_1")
        for _, (z_ref, _) in (z_n << a_m.project(tick=True, rank_id="N")).iterOccupancy(tick=False):
            z_ref += 1
        Metrics.endCollect()

        corr_r0 = [
            "M_pos,M,fiber_pos\n",
            "2,1,0\n",
            "3,2,1\n",
            "7,3,6\n",
            "8,2,1\n",
            "9,1,0\n",
            "10,0,5\n"
        ]

        corr_w0 = [
            "M_pos,M,fiber_pos\n",
            "1,0,5\n",
            "4,2,1\n",
            "6,3,6\n",
            "7,3,3\n",
            "8,2,2\n",
            "9,1,1\n",
            "10,0,0\n"
        ]

        corr_r1 = [
            "M_pos,M,fiber_pos\n",
            "0,0,0\n",
            "2,2,1\n",
            "5,3,2\n"
        ]

        with open("tmp/test_lshift_metrics_fiber_project-N-populate_read_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r0)

        with open("tmp/test_lshift_metrics_fiber_project-N-populate_write_0.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_w0)

        with open("tmp/test_lshift_metrics_fiber_project-N-populate_1.csv", "r") as f:
            self.assertEqual(f.readlines(), corr_r1)

    def test_lshift_use_stats_1D(self):
        """Test reuse statistics collected on a 1D fiber during Fiber.__lshift__"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_m.getRankAttrs().setId("M")

        z_m = Fiber()
        z_m.getRankAttrs().setId("M")

        Metrics.beginCollect("tmp/test_lshift_use_stats_1D")
        Metrics.trace("M")
        Metrics.registerRank("N")
        for n in range(3):
            Metrics.addUse("N", n * 2, n)
            for _ in z_m << a_m:
                pass
            Metrics.incIter("N")
        Metrics.endCollect()

        corr = [
            "N_pos,M_pos,N,M,fiber_pos\n",
            "0,0,0,0,0\n",
            "0,1,0,2,1\n",
            "0,2,0,3,2\n",
            "1,0,2,0,0\n",
            "1,1,2,2,1\n",
            "1,2,2,3,2\n",
            "2,0,4,0,0\n",
            "2,1,4,2,1\n",
            "2,2,4,3,2\n",
        ]

        with open("tmp/test_lshift_use_stats_1D-M-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)


    def test_lshift_use_stats_2D(self):
        """Test reuse statistics collected on a 2D fiber during Fiber.__lshift__"""
        A_JMN = Tensor.fromUncompressed(rank_ids=["J", "M", "N"], root=[[[1, 0, 3], [0, 0, 0], [7, 8, 0]], [[1, 2, 3], [0, 0, 6], [0, 0, 0]]])
        a_j = A_JMN.getRoot()
        Z_MN = Tensor(rank_ids=["M", "N"])
        z_m = Z_MN.getRoot()

        Metrics.beginCollect("tmp/test_lshift_use_stats_2D")
        Metrics.trace("M")
        Metrics.trace("N")
        for j, a_m in a_j:
            for _, (z_n, a_n) in z_m << a_m:
                for _ in z_n << a_n:
                    pass
        Metrics.endCollect()

        M_corr = [
            "J_pos,M_pos,J,M,fiber_pos\n",
            "0,0,0,0,0\n",
            "0,1,0,2,1\n",
            "1,0,1,0,0\n",
            "1,1,1,1,1\n"
        ]

        with open("tmp/test_lshift_use_stats_2D-M-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), M_corr)

        N_corr = [
            "J_pos,M_pos,N_pos,J,M,N,fiber_pos\n",
            "0,0,0,0,0,0,0\n",
            "0,0,1,0,0,2,1\n",
            "0,1,0,0,2,0,0\n",
            "0,1,1,0,2,1,1\n",
            "1,0,0,1,0,0,0\n",
            "1,0,1,1,0,1,1\n",
            "1,0,2,1,0,2,2\n",
            "1,1,0,1,1,2,0\n"
        ]

        with open("tmp/test_lshift_use_stats_2D-N-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), N_corr)

    def test_lshift_with_format(self):
        """Test that Fiber.__lshift__ obeys the specified format"""
        A_M = Tensor.fromUncompressed(rank_ids=["M"], root=[1, 0, 3, 4, 0])
        a_m = A_M.getRoot()

        # A_M is compressed
        A_M.setFormat("M", "C")

        Z_M = Tensor(rank_ids=["M"])
        z_m = Z_M.getRoot()

        inds = []
        for m, _ in z_m << a_m:
            inds.append(m)
        self.assertEqual(inds, [0, 2, 3])

        # A_M is uncompressed
        A_M.setFormat("M", "U")

        Z_M = Tensor(rank_ids=["M"])
        z_m = Z_M.getRoot()

        inds = []
        for m, _ in z_m << a_m:
            inds.append(m)
        self.assertEqual(inds, [0,  1, 2, 3, 4])

    def test_sub(self):
        """Test Fiber.__sub__"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 0, 0])
        b_k.getRankAttrs().setId("K")

        cc = [0, 3]
        cp = [1, 4]
        for i, (c, p) in enumerate(a_k - b_k):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

        id_ = (a_k - b_k).getRankAttrs().getId()
        self.assertEqual(id_, "K")
        self.assertEqual((a_k - b_k).getActive(), (0, 5))


    def test_mul_int(self):
        """Test __mul__ integers"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber([0, 1, 2, 5], [2, 4, 6, 12])

        with self.subTest("f_in * 2"):
            f_out = f_in * 2
            self.assertEqual(f_ref, f_out)

        with self.subTest("2*f_in"):
            f_out = 2 * f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in *=2"):
            # f_in gets clobbered!
            f_in *= 2
            self.assertEqual(f_ref, f_in)

    def test_mul_payload(self):
        """Test __mul__ payload"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber([0, 1, 2, 5], [2, 4, 6, 12])
        two = Payload(2)

        with self.subTest("f_in * 2"):
            f_out = f_in * two
            self.assertEqual(f_ref, f_out)

        with self.subTest("2*f_in"):
            f_out = two * f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in *=2"):
            # f_in gets clobbered!
            f_in *= two
            self.assertEqual(f_ref, f_in)

    def test_mul_eager_only(self):
        """Test __mul__ eager mode only"""
        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_in._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f_in * 2

    def test_imul_eager_only(self):
        """Test __imul__ eager mode only"""
        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_in._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f_in *= 2

if __name__ == '__main__':
    unittest.main()
