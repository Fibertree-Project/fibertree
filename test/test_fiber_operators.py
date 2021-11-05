""" Tests of fiber operators """

import unittest

from fibertree import Fiber
from fibertree import Metrics
from fibertree import Payload
from fibertree import Tensor

class TestFiberOperators(unittest.TestCase):
    """ Tests of fiber operators """

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


    def test_and_metrics_fiber(self):
        """Test metrics collected during Fiber.__and__ on unowned fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 4, 0, 6])
        b_k = Fiber.fromUncompressed([1, 0, 3, 0, 5, 0])

        Metrics.beginCollect()
        for _ in a_k & b_k:
            pass
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank Unknown": {
                "coordinate_read_tensor0": 3,
                "coordinate_read_tensor1": 4,
                "successful_intersect": 1,
                "attempt_intersect": 4,
                "payload_read_tensor0": 1,
                "payload_read_tensor1": 1
            }}
        )

    def test_and_metrics_tensor(self):
        """Test metrics collected during Fiber.__and__ on unowned fibers"""
        A_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 0, 4, 0, 6])
        B_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 3, 0, 5, 0])

        Metrics.beginCollect()
        for _ in A_K.getRoot() & B_K.getRoot():
            pass
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank K": {
                "coordinate_read_tensor0": 3,
                "coordinate_read_tensor1": 4,
                "successful_intersect": 1,
                "attempt_intersect": 4,
                "payload_read_tensor0": 1,
                "payload_read_tensor1": 1
            }}
        )

    def test_and_use_stats_1D(self):
        """Test reuse statistics collected on a 1D fiber during Fiber.__and__"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 4, 5, 6])
        b_k = Fiber.fromUncompressed([1, 0, 3, 0, 5, 0])

        Metrics.beginCollect()
        for _ in range(5):
            for _ in a_k & b_k:
                pass
        Metrics.endCollect()

        reuses, avg_stat = a_k.getUseStats()
        self.assertEqual(reuses, [2] * 8)
        self.assertEqual(avg_stat, 1)

        reuses, avg_stat = b_k.getUseStats()
        self.assertEqual(reuses, [2] * 8)
        self.assertEqual(avg_stat, 1)

    def test_and_use_stats_2D(self):
        """Test reuse statistics collected on a 2D fiber during Fiber.__and__"""
        a_j = Fiber.fromUncompressed([[1, 0, 3], [0, 0, 6], [0, 8, 9]])
        b_i = Fiber.fromUncompressed([[[0, 2, 3], [0, 0, 0], [7, 8, 0]], [[1, 0, 0], [4, 5, 6], [7, 0, 0]]])

        Metrics.beginCollect()
        for _, b_j in b_i:
            for j, (a_k, b_k) in a_j & b_j:
                for _ in a_k & b_k:
                    pass
        Metrics.endCollect()

        reuses, avg_stat = a_j.getUseStats()
        self.assertEqual(reuses, [2, 3])
        self.assertEqual(avg_stat, 0.8)


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

    def test_lshift_metrics_fiber(self):
        """Test metrics collection on Fiber.__lshift__ from a fiber"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        z_m = Fiber.fromUncompressed([0, 2, 3, 0, 0])

        Metrics.beginCollect()
        for _ in z_m << a_m:
            pass
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank Unknown": {
                "coordinate_read_tensor1": 4,
                "payload_read_tensor1": 3,
                "coord_payload_insert_tensor0": 1,
                "coordinate_read_tensor0": 1,
                "payload_read_tensor0": 1,
                "coord_payload_append_tensor0": 1
            }}
        )

    def test_lshift_metrics_tensor(self):
        """Test metrics collection on Fiber.__lshift__ from a tensor"""
        A_M = Tensor.fromUncompressed(rank_ids=["M"], root=[1, 0, 3, 4, 0])
        a_m = A_M.getRoot()

        Z_M = Tensor.fromUncompressed(rank_ids=["M"], root=[0, 2, 3, 0, 0])
        z_m = Z_M.getRoot()

        Metrics.beginCollect()
        for _ in z_m << a_m:
            pass
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank M": {
                "coordinate_read_tensor1": 4,
                "payload_read_tensor1": 3,
                "coord_payload_insert_tensor0": 1,
                "coordinate_read_tensor0": 1,
                "payload_read_tensor0": 1,
                "coord_payload_append_tensor0": 1
            }}
        )

    def test_lshift_use_stats_1D(self):
        """Test reuse statistics collected on a 1D fiber during Fiber.__lshift__"""
        a_m = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        z_m = Fiber()

        Metrics.beginCollect()
        for _ in range(5):
            for _ in z_m << a_m:
                pass
        Metrics.endCollect()

        reuses, avg_stat = a_m.getUseStats()
        self.assertEqual(reuses, [3] * 12)
        self.assertEqual(avg_stat, 1)

        reuses, avg_stat = z_m.getUseStats()
        self.assertEqual(reuses, [3] * 12)
        self.assertEqual(avg_stat, 1)

    def test_lshift_use_stats_2D(self):
        """Test reuse statistics collected on a 2D fiber during Fiber.__lshift__"""
        a_j = Fiber.fromUncompressed([[[1, 0, 3], [0, 0, 0], [7, 8, 0]], [[1, 2, 3], [0, 0, 6], [0, 8, 0]]])
        z_m = Tensor(rank_ids=["M", "N"]).getRoot()

        Metrics.beginCollect()
        for _, a_m in a_j:
            for m, (z_n, a_n) in z_m << a_m:
                for _ in z_n << a_n:
                    pass
        Metrics.endCollect()

        reuses, avg_stat = z_m.getUseStats()
        self.assertEqual(reuses, [4, 6])
        self.assertEqual(avg_stat, 1.8)

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

if __name__ == '__main__':
    unittest.main()
