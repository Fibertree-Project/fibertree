"""Tests of the Intersector classes"""

import os
import unittest

from fibertree import Fiber
from fibertree import Metrics
from fibertree.model import *

class TestIntersector(unittest.TestCase):
    """Tests of the intersector classes"""

    def test_num_isect_leader_follower(self):
        """Test the LeaderFollowerIntersector"""
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
            a_intersector.addTraces(Metrics.consumeTrace("K", "intersect_0"))
            b_intersector.addTraces(Metrics.consumeTrace("K", "intersect_1"))

        Metrics.endCollect()

        self.assertEqual(a_intersector.getNumIntersects(), 12)
        self.assertEqual(b_intersector.getNumIntersects(), 9)

    def test_num_isect_skip_ahead(self):
        """ Test SkipAheadIntersector"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 0, 0, 0, 0, 8, 4])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 4, 6, 0, 0, 4, 0])
        b_k.getRankAttrs().setId("K")

        intersector = SkipAheadIntersector()

        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)
        for _ in a_k & b_k:
            pass
        intersector.addTraces(
            Metrics.consumeTrace("K", "intersect_0"),
            Metrics.consumeTrace("K", "intersect_1"))
        Metrics.endCollect()

        self.assertEqual(intersector.getNumIntersects(), 3)

    def test_num_isect_cannot_skip_across_fibers(self):
        """Test SkipAheadIntersector does not skip across fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 0, 5])
        b_k.getRankAttrs().setId("K")
        c_j = Fiber.fromUncompressed([1, 2, 3])
        c_j.getRankAttrs().setId("J")

        intersector = SkipAheadIntersector()

        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)
        for _ in c_j:
            for _ in a_k & b_k:
                pass
            intersector.addTraces(
                Metrics.consumeTrace("K", "intersect_0"),
                Metrics.consumeTrace("K", "intersect_1"))
        Metrics.endCollect()

        self.assertEqual(intersector.getNumIntersects(), 4 * 3)

    def test_num_isect_cannot_skip_across_fibers_one_shot(self):
        """Test SkipAheadIntersector does not skip across fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 0, 5])
        b_k.getRankAttrs().setId("K")
        c_j = Fiber.fromUncompressed([1, 2, 3])
        c_j.getRankAttrs().setId("J")

        intersector = SkipAheadIntersector()

        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)
        for _ in c_j:
            for _ in a_k & b_k:
                pass
        intersector.addTraces(
            Metrics.consumeTrace("K", "intersect_0"),
            Metrics.consumeTrace("K", "intersect_1"))
        Metrics.endCollect()

        self.assertEqual(intersector.getNumIntersects(), 4 * 3)

    def test_num_isect_two_finger(self):
        """ Test TwoFingerIntersector"""
        a_k = Fiber.fromUncompressed([1, 0, 0, 0, 0, 0, 0, 8, 4])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 4, 6, 0, 0, 4, 0])
        b_k.getRankAttrs().setId("K")

        intersector = TwoFingerIntersector()

        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)
        for _ in a_k & b_k:
            pass
        intersector.addTraces(
            Metrics.consumeTrace("K", "intersect_0"),
            Metrics.consumeTrace("K", "intersect_1"))
        Metrics.endCollect()

        self.assertEqual(intersector.getNumIntersects(), 6)

    def test_num_isect_two_finger_cannot_compare_across_fibers(self):
        """ Test TwoFingerIntersector does not compare across fibers"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 0])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 2, 3, 0, 5])
        b_k.getRankAttrs().setId("K")
        c_j = Fiber.fromUncompressed([1, 2, 3])
        c_j.getRankAttrs().setId("J")

        intersector = TwoFingerIntersector()

        Metrics.beginCollect()
        Metrics.trace("K", "intersect_0", consumable=True)
        Metrics.trace("K", "intersect_1", consumable=True)

        for _ in c_j:
            for _ in a_k & b_k:
                pass
            intersector.addTraces(
                Metrics.consumeTrace("K", "intersect_0"),
                Metrics.consumeTrace("K", "intersect_1"))
        Metrics.endCollect()

        self.assertEqual(intersector.getNumIntersects(), 4 * 3)
