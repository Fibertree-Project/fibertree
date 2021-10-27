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
        _ = a_k & b_k
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank Unknown": {
                "metadata_read_tensor0": 3,
                "metadata_read_tensor1": 4,
                "data_read_tensor0": 1,
                "data_read_tensor1": 1,
                "successful_intersect": 1,
                "attempt_intersect": 4
            }}
        )


    def test_and_metrics_1d_2d_tensor(self):
        """Test metrics collected during Fiber.__and__ on a 1D and 2D tensor"""
        A_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 0, 4, 0, 6])
        a_k = A_K.getRoot()

        B_KM = Tensor.fromUncompressed(
                rank_ids=["K", "M"],
                root=[[1], [0], [3], [0], [5], [0]]
        )
        b_k = B_KM.getRoot()

        Metrics.beginCollect()
        _ = a_k & b_k
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank K": {
                "metadata_read_tensor0": 3,
                "metadata_read_tensor1": 4,
                "data_read_tensor0": 1,
                "successful_intersect": 1,
                "attempt_intersect": 4
            }}
        )


    def test_and_metrics_2d_1d_tensor(self):
        """Test metrics collected during Fiber.__and__ on a 2D and 1D tensor"""
        A_KM = Tensor.fromUncompressed(
                rank_ids=["K", "M"],
                root=[[1], [0], [0], [4], [0], [6]])
        a_k = A_KM.getRoot()

        B_K = Tensor.fromUncompressed(rank_ids=["K"], root=[1, 0, 3, 0, 5, 0])
        b_k = B_K.getRoot()

        Metrics.beginCollect()
        _ = a_k & b_k
        Metrics.endCollect()

        self.assertEqual(
            Metrics.dump(),
            {"Rank K": {
                "metadata_read_tensor0": 3,
                "metadata_read_tensor1": 4,
                "data_read_tensor1": 1,
                "successful_intersect": 1,
                "attempt_intersect": 4
            }}
        )


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
