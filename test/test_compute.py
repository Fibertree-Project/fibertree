"""Tests of the Compute class"""

import unittest

from fibertree import Metrics
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor
from fibertree.model import Compute

class TestCompute(unittest.TestCase):
    """Tests of the Compute class"""

    def test_compute_op(self):
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)

        Metrics.beginCollect("", ["K"])
        _ = a * b
        self.assertEqual(Compute.opCount(Metrics.dump(), "mul"), 1)
        self.assertEqual(Compute.opCount(Metrics.dump(), "add"), 0)

        _ = a + 2
        _ = 1 * b

        a *= b
        self.assertEqual(Compute.opCount(Metrics.dump(), "mul"), 3)
        self.assertEqual(Compute.opCount(Metrics.dump(), "add"), 1)

        Metrics.endCollect()

    def test_compute_lf(self):
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 5])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 0, 6, 7, 9])
        b_k.getRankAttrs().setId("K")

        Metrics.beginCollect("", ["M", "K"])
        for _ in range(1):
            for _ in Fiber.intersection(a_k, b_k):
                pass
            Metrics.incIter("M")
        Metrics.endCollect()

        self.assertEqual(Compute.lfCount(Metrics.dump(), "K", 0), 4)
        self.assertEqual(Compute.lfCount(Metrics.dump(), "K", 1), 3)

    def test_compute_skip(self):
        a_k = Fiber.fromUncompressed([1, 0, 0, 0, 0, 0, 0, 8, 4])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 4, 6, 0, 0, 4, 0])
        b_k.getRankAttrs().setId("K")

        Metrics.beginCollect("", ["M", "K"])
        for _ in range(1):
            for _ in Fiber.intersection(a_k, b_k):
                pass
            Metrics.incIter("M")
        Metrics.endCollect()

        self.assertEqual(Compute.skipCount(Metrics.dump(), "K"), 3)

    def test_compute_swaps_discrete_next(self):
        """Test swapCount"""
        coords = [0, 1]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2, 3], [1, 1, 1])]
        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))

        self.assertEqual(Compute.swapCount(tensor, 0, 100, 3), 3 * (2 + 6))

    def test_compute_swaps_finite_radix(self):
        """Test swapCount when limited by the radix"""
        coords = [0, 1, 2]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2, 3], [1, 1, 1]),
                    Fiber([0, 4], [1, 1])]

        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))
        ops = 3 * (2 + 6) + 3 * (1 + 2) + 3 * (2 + 8)

        self.assertEqual(Compute.swapCount(tensor, 0, 2, 3), ops)

    def test_compute_swaps_undefined_next(self):
        """Test swapCount when the next_latency is N"""
        coords = [0, 1, 2]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([0, 2, 3], [1, 1, 1]),
                    Fiber([1, 4], [1, 1])]

        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))
        ops = (1 + 1 + 2) + (3 + 3 + 2 + 1 + 0 + 2 + 0 + 0)

        self.assertEqual(Compute.swapCount(tensor, 0, float("inf"), "N"), ops)

    def test_compute_swaps_depth_1(self):
        """Test swapCount when the depth is > 0"""
        coords_0 = [0, 1, 2]
        payloads_0 = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([0, 2, 3], [1, 1, 1]),
                    Fiber([1, 4], [1, 1])]
        fiber_0 = Fiber(coords_0, payloads_0)

        coords_1 = [0, 1]
        payloads_1 = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2, 3], [1, 1, 1])]
        fiber_1 = Fiber(coords_1, payloads_1)

        root = Fiber([3, 7], [fiber_0, fiber_1])
        tensor = Tensor.fromFiber(rank_ids=["M", "N", "K"], fiber=root)
        ops = 3 * (2 + 6) + 3 * (1 + 2) + 3 * (2 + 8) + 3 * (2 + 6)

        self.assertEqual(Compute.swapCount(tensor, 1, 2, 3), ops)


if __name__ == '__main__':
    unittest.main()
