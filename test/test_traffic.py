"""Tests of the Traffic class"""

import unittest
import yaml

from fibertree import Metrics, Tensor
from fibertree.model import Format, Traffic

class TestTraffic(unittest.TestCase):
    """Tests of the Traffic class"""

    def setUp(self):
        K = 8
        M = 6
        N = 7
        density = 0.5

        # Create the tensors
        A_KM = Tensor.fromRandom(
            rank_ids=[
                "K", "M"], shape=[
                K, M], density=[
                    0.9, density], seed=0)
        self.B_KN = Tensor.fromRandom(
            rank_ids=[
                "K", "N"], shape=[
                K, N], density=[
                    0.9, density], seed=1)
        self.A_MK = A_KM.swizzleRanks(rank_ids=["M", "K"])

        b_k = self.B_KN.getRoot()
        a_m = self.A_MK.getRoot()
        T_MKN = Tensor(rank_ids=["M", "K", "N"])
        t_m = T_MKN.getRoot()

        Metrics.beginCollect("tmp/test_traffic_stage0", ["M", "K", "N"])
        Metrics.traceRank("K")
        for m, (t_k, a_k) in t_m << a_m:
            for k, (t_n, (a_val, b_n)) in t_k << (a_k & b_k):
                for n, (t_ref, b_val) in t_n << b_n:
                    t_ref += b_val
        Metrics.endCollect()

        a_m = self.A_MK.getRoot()
        T_MNK = T_MKN.swizzleRanks(rank_ids=["M", "N", "K"])
        t_m = T_MNK.getRoot()
        self.Z_MN = Tensor(rank_ids=["M", "N"])
        z_m = self.Z_MN.getRoot()

        Metrics.beginCollect("tmp/test_traffic_stage1", ["M", "N", "K"])
        for m, (z_n, (t_n, a_k)) in z_m << (t_m & a_m):
            for n, (z_ref, t_k) in z_n << t_n:
                for k, (t_val, a_val) in t_k & a_k:
                    z_ref += t_val * a_val
        Metrics.endCollect()

        formats = yaml.safe_load("""
        B:
            K:
                format: U
                rhbits: 32
                pbits: 32
            N:
                format: C
                cbits: 32
                pbits: 64
        """)
        self.B_format = Format(self.B_KN, formats["B"])

    # TODO: Fix stats collection and retest
    # def test_buffetTraffic(self):
    #     """Test buffetTraffic"""
    #     bytes_ = Traffic.buffetTraffic(self.B_KN, "K", self.B_format)
    #     corr = 480 + 288 + 288 + 480 + 480 + 288 + 288 + 96 + 480 + 96 + 288 + 288 + 288 + 288
    #     self.assertEqual(bytes_, corr)

    # def test_cacheTraffic(self):
    #     """Test cacheTraffic"""
    #     bytes_ = Traffic.cacheTraffic(self.B_KN, "K", self.B_format, 2**10)
    #     corr = 480 + 288 + 288 + 480 +  0 + 288 + 288 + 96 + 0 + 0 + 288 + 288 + 0 + 0
    #     self.assertEqual(bytes_, corr)


    def test_getAllUses(self):
        """Test _getAllUses"""
        reuses = {(1, 2): {3: ((0, 0, 0), [(4, 3, 2)]), 6: ((0, 0, 1), [])},
                  (10, 8): {7: ((0, 5, 6), [(2, 1, 3), (3, 1, 2)])}}

        uses = [((0, 0, 0), (1, 2, 3)), ((4, 3, 2), (1, 2, 3)),
                ((0, 0, 1), (1, 2, 6)), ((0, 5, 6), (10, 8, 7)),
                ((2, 6, 9), (10, 8, 7)), ((3, 6, 8), (10, 8, 7))]

        assert Traffic._getAllUses(reuses) == uses

    def test_getUse(self):
        """Test _getUse"""
        self.assertEqual(Traffic._getUse((1, 2, 3), (4, 5, 3)), (5, 7, 6))

    def test_optimalEvict(self):
        """Test _optimalEvict"""
        uses = [1, 3, 5, 1, 7, 4, 5]
        use_data = {1: [4, [3, 0]], 3: [12, [1]], 5: [
            20, [6, 2]], 7: [28, [4]], 4: [16, [5]]}
        objs = [5, 4, 1]
        assert Traffic._optimalEvict(use_data, objs) == 4
