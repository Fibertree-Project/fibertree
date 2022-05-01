"""Tests of the Compute class"""

import unittest

from fibertree import Metrics
from fibertree import Payload
from fibertree import Fiber
from fibertree.model.compute import Compute

class TestCompute(unittest.TestCase):
    """Tests of the Compute class"""

    def test_compute_op(self):
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)

        Metrics.beginCollect(["K"])
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

        Metrics.beginCollect(["M", "K"])
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

        Metrics.beginCollect(["M", "K"])
        for _ in range(1):
            for _ in Fiber.intersection(a_k, b_k):
                pass
            Metrics.incIter("M")
        Metrics.endCollect()
        
        self.assertEqual(Compute.skipCount(Metrics.dump(), "K"), 3)

if __name__ == '__main__':
    unittest.main()