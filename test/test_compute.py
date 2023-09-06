"""Tests of the Compute class"""

import os
import unittest

from fibertree import Metrics
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor
from fibertree.model import Compute

class TestCompute(unittest.TestCase):
    """Tests of the Compute class"""

    def setUp(self):
        # Make sure we have a tmp directory to write to
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def test_num_iters(self):
        """Test Compute.numIters()"""

        coords = [0, 1]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2], [1, 1])]
        T_MK = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))
        t_m = T_MK.getRoot()

        Metrics.beginCollect("tmp/test_num_iters")
        Metrics.trace("K")
        for m, t_k in t_m:
            for _ in t_k:
                pass
        Metrics.endCollect()

        self.assertEqual(Compute.numIters("tmp/test_num_iters-K-iter.csv"), 5)

    def test_num_ops(self):
        """Test Compute.numOps()"""
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)

        Metrics.beginCollect()
        _ = a * b
        self.assertEqual(Compute.numOps(Metrics.dump(), "mul"), 1)
        self.assertEqual(Compute.numOps(Metrics.dump(), "add"), 0)

        _ = a + 2
        _ = 1 * b

        a *= b
        self.assertEqual(Compute.numOps(Metrics.dump(), "mul"), 3)
        self.assertEqual(Compute.numOps(Metrics.dump(), "add"), 1)

        Metrics.endCollect()

    def test_num_swaps_discrete_next(self):
        """Test Compute.numSwaps()"""
        coords = [0, 1]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2, 3], [1, 1, 1])]
        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))

        self.assertEqual(Compute.numSwaps(tensor, 0, 100, 3), 3 * (2 + 6))

    def test_num_swaps_finite_radix(self):
        """Test Compute.numSwaps() when limited by the radix"""
        coords = [0, 1, 2]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([1, 2, 3], [1, 1, 1]),
                    Fiber([0, 4], [1, 1])]

        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))
        ops = 3 * (2 + 6) + 3 * (1 + 2) + 3 * (2 + 8)

        self.assertEqual(Compute.numSwaps(tensor, 0, 2, 3), ops)

    def test_num_swaps_undefined_next(self):
        """Test Compute.numSwaps when the next_latency is N"""
        coords = [0, 1, 2]
        payloads = [Fiber([1, 3, 5], [1, 1, 1]), Fiber([0, 2, 3], [1, 1, 1]),
                    Fiber([1, 4], [1, 1])]

        tensor = Tensor.fromFiber(rank_ids=["M", "K"], fiber=Fiber(coords, payloads))
        ops = (1 + 1 + 2) + (3 + 3 + 2 + 1 + 0 + 2 + 0 + 0)

        self.assertEqual(Compute.numSwaps(tensor, 0, float("inf"), "N"), ops)

    def test_num_swaps_depth_1(self):
        """Test Compute.numSwaps when the depth is > 0"""
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

        self.assertEqual(Compute.numSwaps(tensor, 1, 2, 3), ops)


if __name__ == '__main__':
    unittest.main()
