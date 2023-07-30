"""Tests of the Intersector classes"""

import os
import unittest

from fibertree import Fiber
from fibertree import Metrics
from fibertree.model import *

class TestIntersector(unittest.TestCase):
    """Tests of the intersector classes"""

    def test_num_isect_leader_follower(self):
        """Test Compute.numIsectLeaderFollower()"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 5])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 0, 6, 7, 9])
        b_k.getRankAttrs().setId("K")

        a_intersector = LeaderFollowerIntersector()
        b_intersector = LeaderFollowerIntersector()


        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)
        for i in range(3):
            for _ in a_k & b_k:
                pass
            a_intersector.addTrace(Metrics.consumeTrace("K", "intersect_0"))
            b_intersector.addTrace(Metrics.consumeTrace("K", "intersect_1"))

        Metrics.endCollect()

        self.assertEqual(a_intersector.getNumIntersects(), 12)
        self.assertEqual(b_intersector.getNumIntersects(), 9)

