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
            Metrics.addUse("K", 4)

    def test_add_use_none_traced(self):
        """Test that addUse only adds a file if the rank is traced"""
        Metrics.beginCollect("tmp/test_add_use_none_traced")
        Metrics.registerRank("K")
        Metrics.addUse("K", 2)
        Metrics.endCollect()

        self.assertFalse(os.path.exists("tmp/test_add_use_none_traced-K.csv"))

    def test_add_use_one_traced(self):
        """Test that addUse correctly traces a rank if it is being traced"""
        Metrics.beginCollect("tmp/test_add_use_one_traced")
        Metrics.traceRank("K")

        ks = [[3, 7], [8]]

        Metrics.registerRank("M")
        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m)
            Metrics.registerRank("K")
            for k in ks[i]:
                Metrics.addUse("K", k)
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")
            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K\n",
            "0,0,2,3\n",
            "0,1,2,7\n",
            "1,0,5,8\n"
        ]

        with open("tmp/test_add_use_one_traced-K.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_num_cached_uses(self):
        """Test that num_cached_uses is followed"""
        Metrics.beginCollect("tmp/test_add_use_num_cached_uses")
        Metrics.setNumCachedUses(2)
        Metrics.traceRank("K")

        ks = [[3, 7], [1, 8]]

        corr = [
            "M_pos,K_pos,M,K\n",
            "0,0,2,3\n",
            "0,1,2,7\n",
            "1,0,5,1\n",
            "1,1,5,8\n"
        ]

        Metrics.registerRank("M")
        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m)
            Metrics.registerRank("K")
            for k in ks[i]:
                Metrics.addUse("K", k)
                Metrics.registerRank("N")
                for n in range(3):
                    Metrics.addUse("N", n)
                    Metrics.incIter("N")
                Metrics.endIter("N")
                Metrics.incIter("K")

            with open("tmp/test_add_use_num_cached_uses-K.csv", "r") as f:
                self.assertEqual(f.readlines(), corr[:(2 * i + 2)])

            Metrics.endIter("K")
            Metrics.incIter("M")
        Metrics.endIter("M")

        Metrics.endCollect()

        with open("tmp/test_add_use_num_cached_uses-K.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_end_iter_fails_if_not_collecting(self):
        """Test that endIter fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.endIter("K")


    def test_empty_dump(self):
        """Test that if no metrics have been collected, the dump is empty"""
        Metrics.beginCollect()
        Metrics.endCollect()

        self.assertEqual(Metrics.dump(), {})

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
        self.assertEqual(Metrics.getIter(), (0, 0))

        Metrics.incIter("M")
        self.assertEqual(Metrics.getIter(), (0, 1))

        Metrics.incIter("N")
        Metrics.incIter("N")
        Metrics.incIter("N")
        self.assertEqual(Metrics.getIter(), (3, 1))

        Metrics.endIter("M")
        self.assertEqual(Metrics.getIter(), (3, 0))

        Metrics.endCollect()

    def test_end_iter_without_inc(self):
        """Test that endIter functions correctly even if the corresponding
        iterator has not yet been incremented"""
        Metrics.beginCollect()
        Metrics.registerRank("N")
        Metrics.registerRank("M")

        Metrics.endIter("M")
        Metrics.incIter("N")

        self.assertEqual(Metrics.getIter(), (1, 0))

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

    def test_register_rank_fails_if_not_collecting(self):
        """Test that registerRank fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.registerRank("K")

    def test_trace_rank_fails_if_not_collecting(self):
        """Test that traceRank fails if collection is not on"""
        with self.assertRaises(AssertionError):
            Metrics.traceRank("K")

    def test_trace_rank_has_prefix(self):
        """Test that a prefix has been specified if we want to trace a rank"""

        Metrics.beginCollect()

        with self.assertRaises(AssertionError):
            Metrics.traceRank("M")

        Metrics.endCollect()
