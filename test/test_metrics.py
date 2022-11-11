import os
import unittest

from fibertree import *

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

        # Make sure we have a tmp directory to write to
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def test_is_collecting(self):
        """Test the isCollecting() method directly"""

        Metrics.collecting = True
        self.assertTrue(Metrics.isCollecting())

        Metrics.collecting = False
        self.assertFalse(Metrics.isCollecting())

    def test_begin_collect(self):
        """Test that the beginCollect() method begins collection"""
        self.assertFalse(Metrics.isCollecting())
        Metrics.beginCollect()
        self.assertTrue(Metrics.isCollecting())

        Metrics.endCollect()

    def test_end_collect(self):
        """Test that the endCollect() method ends collection"""
        Metrics.beginCollect()

        self.assertTrue(Metrics.isCollecting())
        Metrics.endCollect()
        self.assertFalse(Metrics.isCollecting())

    def test_add_use_fails_if_not_collecting(self):
        """Test that addUse fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.addUse("K", 4, 1)

    def test_add_use_none_traced(self):
        """Test that addUse only adds a file if the rank is traced"""
        Metrics.beginCollect("tmp/test_add_use_none_traced")
        Metrics.registerRank("K")
        Metrics.addUse("K", 2, 1)
        Metrics.endCollect()

        self.assertFalse(os.path.exists("tmp/test_add_use_none_traced-K-iter.csv"))

    def test_add_use_one_traced(self):
        """Test that addUse correctly traces a rank if it is being traced"""
        Metrics.beginCollect("tmp/test_add_use_one_traced")
        Metrics.trace("K")

        ks = [[3, 7], [8]]

        Metrics.registerRank("M")
        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m, i)
            Metrics.registerRank("K")
            for j, k in enumerate(ks[i]):
                Metrics.addUse("K", k, 2 * j + 1)
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n, n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")
            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "0,0,2,3,1\n",
            "0,1,2,7,3\n",
            "1,0,5,8,1\n"
        ]

        with open("tmp/test_add_use_one_traced-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_num_cached_uses(self):
        """Test that num_cached_uses is followed"""
        Metrics.beginCollect("tmp/test_add_use_num_cached_uses")
        Metrics.setNumCachedUses(2)
        Metrics.trace("K")

        ks = [[3, 7], [1, 8]]

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "0,0,2,3,10\n",
            "0,1,2,7,14\n",
            "1,0,5,1,8\n",
            "1,1,5,8,15\n"
        ]

        Metrics.registerRank("M")
        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m, i)
            Metrics.registerRank("K")
            for k in ks[i]:
                Metrics.addUse("K", k, k + 7)
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n, n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")

            with open("tmp/test_add_use_num_cached_uses-K-iter.csv", "r") as f:
                self.assertEqual(f.readlines(), corr[:(2 * i + 2)])

            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        with open("tmp/test_add_use_num_cached_uses-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_other_type(self):
        """Test that addUse works with types other than "iter" """
        Metrics.beginCollect("tmp/test_add_use_other_type")
        Metrics.trace("K", type_="other")

        ks = [[3, 7], [8]]

        Metrics.registerRank("M")
        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m, i)
            Metrics.registerRank("K")
            for k in ks[i]:
                Metrics.addUse("K", k, k - 1, type_="other")
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n, n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")
            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "0,0,2,3,2\n",
            "0,1,2,7,6\n",
            "1,0,5,8,7\n"
        ]

        with open("tmp/test_add_use_other_type-K-other.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_explicit_iteration(self):
        """Add use works with an explicit iteration stamp"""
        Metrics.beginCollect("tmp/test_add_use_explicit_iteration")
        Metrics.trace("K")
        Metrics.registerRank("M")

        ks = [[3, 7], [8]]

        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m, i)
            Metrics.registerRank("K")
            for k in ks[i]:
                Metrics.addUse("K", k, k - 1, iteration_num=[i + 1, k + 2])
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n, n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")
            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "1,5,2,3,2\n",
            "1,9,2,7,6\n",
            "2,10,5,8,7\n"
        ]

        with open("tmp/test_add_use_explicit_iteration-K-iter.csv") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_matched_ranks(self):
        Metrics.beginCollect("tmp/test_add_use_matched_ranks")
        Metrics.trace("K", "matched")
        Metrics.registerRank("N")
        Metrics.registerRank("M")
        Metrics.matchRanks("K", "M")
        for i, n in enumerate([2, 4]):
            Metrics.addUse("N", n, i)
            for j, m in enumerate([7, 9, 11]):
                Metrics.addUse("M", m, j)
                Metrics.addUse("K", m - 3, j + 1, type_="matched")
                Metrics.incIter("M")
            Metrics.endIter("M")
            Metrics.incIter("N")
        Metrics.endIter("N")
        Metrics.endCollect()

        corr = [
            "N_pos,M_pos,N,M,fiber_pos\n",
            "0,0,2,4,1\n",
            "0,1,2,6,2\n",
            "0,2,2,8,3\n",
            "1,0,4,4,1\n",
            "1,1,4,6,2\n",
            "1,2,4,8,3\n"
        ]


    def test_end_iter_fails_if_not_collecting(self):
        """Test that endIter fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.endIter("K")

    def test_empty_dump(self):
        """Test that if no metrics have been collected, the dump is empty"""
        Metrics.beginCollect()
        Metrics.endCollect()

        self.assertEqual(Metrics.dump(), {})

    def test_getIndex(self):
        """Test that getIndex works"""
        Metrics.beginCollect()
        Metrics.registerRank("M")
        Metrics.registerRank("N")

        self.assertEqual(Metrics.getIndex("M"), 0)
        self.assertEqual(Metrics.getIndex("N"), 1)

        Metrics.matchRanks("K", "M")

        self.assertEqual(Metrics.getIndex("K"), 0)

        with self.assertRaises(AssertionError):
            Metrics.getIndex("J")

        Metrics.endCollect()

    def test_inc_count_fails_if_not_collecting(self):
        """Test that incCount fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.incCount("Line 1", "Metric 4", 2)

    def test_inc_count(self):
        """Test that inc updates the correct line/metric and adds new entries
        to the metrics dictionary if the line/metric do not already exist"""
        Metrics.beginCollect()
        Metrics.incCount("Line 1", "Metric 1", 5)
        self.assertEqual(Metrics.dump(), {"Line 1": {"Metric 1": 5}})

        Metrics.incCount("Line 1", "Metric 1", 6)
        self.assertEqual(Metrics.dump(), {"Line 1": {"Metric 1": 11}})

        Metrics.incCount("Line 1", "Metric 2", 4)
        self.assertEqual(
            Metrics.dump(),
            {"Line 1": {"Metric 1": 11, "Metric 2": 4}}
        )

        Metrics.incCount("Line 2", "Metric 1", 7)
        self.assertEqual(
            Metrics.dump(),
            {"Line 1": {"Metric 1": 11, "Metric 2": 4},
             "Line 2": {"Metric 1": 7}
            }
        )

        Metrics.endCollect()

    def test_inc_iter_fails_if_not_collecting(self):
        """Test that incIter fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.incIter("K")

    def test_inc_iter(self):
        """Test that the iterator increments correctly"""
        Metrics.beginCollect()
        Metrics.registerRank("N")
        Metrics.registerRank("M")
        self.assertEqual(Metrics.getIter(), [0, 0])

        Metrics.incIter("M")
        self.assertEqual(Metrics.getIter(), [0, 1])

        Metrics.incIter("N")
        Metrics.incIter("N")
        Metrics.incIter("N")
        self.assertEqual(Metrics.getIter(), [3, 1])

        Metrics.endIter("M")
        self.assertEqual(Metrics.getIter(), [3, 0])

        Metrics.endCollect()

    def test_end_iter_without_inc(self):
        """Test that endIter functions correctly even if the corresponding
        iterator has not yet been incremented"""
        Metrics.beginCollect()
        Metrics.registerRank("N")
        Metrics.registerRank("M")

        Metrics.endIter("M")
        Metrics.incIter("N")

        self.assertEqual(Metrics.getIter(), [1, 0])

        Metrics.endCollect()

    def test_new_collection(self):
        """Test that a beginCollect() restarts collection"""
        Metrics.beginCollect()
        Metrics.incCount("Line 1", "Metric 1", 5)
        Metrics.endCollect()

        Metrics.beginCollect()
        Metrics.endCollect()
        self.assertEqual(Metrics.dump(), {})

    def test_get_label_fails_if_not_collecting(self):
        """Test that getLabel fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.getLabel("K")

    def tesst_get_label_fails_if_unknown_rank(self):
        """Test that getLabel fails if the rank is unknown"""
        Metrics.beginCollect()

        with self.assertRaises(AssertionError):
            Metrics.getLabel("K")

        Metrics.endCollect()

    def test_get_label(self):
        """Test that getLabel correctly labels tensors"""
        Metrics.beginCollect()
        Metrics.registerRank("K")

        self.assertEqual(Metrics.getLabel("K"), 0)
        self.assertEqual(Metrics.getLabel("K"), 1)
        self.assertEqual(Metrics.getLabel("K"), 2)

        Metrics.endIter("K")

        self.assertEqual(Metrics.getLabel("K"), 0)

        Metrics.endCollect()

    def test_match_ranks(self):
        """Test that ranks are matched correctly"""
        Metrics.beginCollect("tmp/test_match_ranks")
        Metrics.trace("M", "match_ranks")

        # At least one must be in the loop order
        with self.assertRaises(AssertionError):
            Metrics.matchRanks("K", "M")

        # Either rank can be in either position
        Metrics.registerRank("K")
        Metrics.matchRanks("K", "M")
        Metrics.matchRanks("N", "K")

        # Metrics.getLabel still works correctly
        self.assertEqual(Metrics.getLabel("K"), 0)
        self.assertEqual(Metrics.getLabel("M"), 1)
        self.assertEqual(Metrics.getLabel("N"), 2)

        # Metrics.addUse() still works correctly
        Metrics.addUse("M", 5, 2, type_="match_ranks")
        Metrics.addUse("K", 2, 1, type_="iter")
        Metrics.incIter("K")
        Metrics.addUse("M", 4, 5, type_="match_ranks")
        Metrics.endIter("K")
        Metrics.addUse("M", 8, 7, type_="match_ranks")

        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,5,2\n",
            "1,4,5\n",
            "0,8,7\n"
        ]

        with open("tmp/test_match_ranks-M-match_ranks.csv") as f:
            self.assertEqual(f.readlines(), corr)

    def test_match_rank_unmatched_label(self):
        """Test that match rank correctly combines labels"""
        Metrics.beginCollect()

        self.assertEqual(Metrics.getLabel("K"), 0)
        self.assertEqual(Metrics.getLabel("K"), 1)

        Metrics.registerRank("M")
        Metrics.matchRanks("M", "K")

        self.assertEqual(Metrics.getLabel("M"), 2)

        Metrics.endCollect()

    def test_register_rank_fails_if_not_collecting(self):
        """Test that registerRank fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.registerRank("K")

    def test_trace_rank_fails_if_not_collecting(self):
        """Test that trace fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.trace("K")

    def test_trace_rank_has_prefix(self):
        """Test that a prefix has been specified if we want to trace a rank"""

        Metrics.beginCollect()

        with self.assertRaises(AssertionError):
            Metrics.trace("M")

        Metrics.endCollect()
