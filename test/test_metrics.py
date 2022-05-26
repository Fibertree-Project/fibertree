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
        Metrics.beginCollect("", ["K"])
        self.assertTrue(Metrics.isCollecting())

        Metrics.endCollect()

    def test_end_collect(self):
        """Test that the endCollect() method ends collection"""
        Metrics.beginCollect("", ["K"])

        self.assertTrue(Metrics.isCollecting())
        Metrics.endCollect()
        self.assertFalse(Metrics.isCollecting())

    def test_add_use_none_traced(self):
        """Test that addUse only adds a file if the rank is traced"""
        Metrics.beginCollect("tmp/test_add_use_none_traced", ["K"])
        Metrics.addUse("K", 2)
        Metrics.endCollect()

        self.assertFalse(os.path.exists("tmp/test_add_use_none_traced-K.csv"))

    def test_add_use_one_traced(self):
        """Test that addUse correctly traces a rank if it is being traced"""
        Metrics.beginCollect("tmp/test_add_use_one_traced", ["M", "K", "N"])
        Metrics.traceRank("K")

        ks = [[3, 7], [8]]

        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m)
            for k in ks[i]:
                Metrics.addUse("K", k)
                for n in range(3):
                    Metrics.addUse("N", n)
                    Metrics.incIter("N")
                Metrics.clrIter("N")
                Metrics.incIter("K")
            Metrics.clrIter("K")
            Metrics.incIter("M")
        Metrics.clrIter("M")

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
        Metrics.beginCollect("tmp/test_add_use_num_cached_uses", ["M", "K", "N"])
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

        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m)
            for k in ks[i]:
                Metrics.addUse("K", k)
                for n in range(3):
                    Metrics.addUse("N", n)
                    Metrics.incIter("N")
                Metrics.clrIter("N")
                Metrics.incIter("K")

            with open("tmp/test_add_use_num_cached_uses-K.csv", "r") as f:
                self.assertEqual(f.readlines(), corr[:(2 * i + 2)])

            Metrics.clrIter("K")
            Metrics.incIter("M")
        Metrics.clrIter("M")

        Metrics.endCollect()

        with open("tmp/test_add_use_num_cached_uses-K.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_add_use_repeat_iter(self):
        """Test that addUse correctly removes repeated iteration"""
        Metrics.beginCollect("tmp/test_add_use_repeat_iter", ["M", "K", "N"])
        Metrics.traceRank("K")

        ks = [[3, 7], [8]]

        for i, m in enumerate([2, 5]):
            Metrics.addUse("M", m)
            for k in ks[i]:
                Metrics.addUse("K", k)
                Metrics.addUse("K", k + 1)
                for n in range(3):
                    Metrics.addUse("N", n)
                    Metrics.incIter("N")
                Metrics.clrIter("N")
                Metrics.incIter("K")
            Metrics.clrIter("K")
            Metrics.incIter("M")
        Metrics.clrIter("M")

        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K\n",
            "0,0,2,4\n",
            "0,1,2,8\n",
            "1,0,5,9\n"
        ]

        with open("tmp/test_add_use_repeat_iter-K.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_empty_dump(self):
        """Test that if no metrics have been collected, the dump is empty"""
        Metrics.beginCollect("", ["K"])
        Metrics.endCollect()

        self.assertEqual(Metrics.dump(), {})

    def test_inc_count(self):
        """Test that inc updates the correct line/metric and adds new entries
        to the metrics dictionary if the line/metric do not already exist"""
        Metrics.beginCollect("", ["K"])
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

    def test_inc_iter(self):
        """Test that the iterator increments correctly"""
        Metrics.beginCollect("", ["N", "M"])
        self.assertEqual(Metrics.getIter(), (0, 0))

        Metrics.incIter("M")
        self.assertEqual(Metrics.getIter(), (0, 1))

        Metrics.incIter("N")
        Metrics.incIter("N")
        Metrics.incIter("N")
        self.assertEqual(Metrics.getIter(), (3, 1))

        Metrics.clrIter("M")
        self.assertEqual(Metrics.getIter(), (3, 0))

        Metrics.endCollect()

    def test_clr_iter_without_inc(self):
        """Test that clear functions correctly even if the corresponding
        iterator has not yet been incremented"""
        Metrics.beginCollect("", ["N", "M"])

        Metrics.clrIter("M")
        Metrics.incIter("N")

        self.assertEqual(Metrics.getIter(), (1, 0))

        Metrics.endCollect()

    def test_new_collection(self):
        """Test that a beginCollect() restarts collection"""
        Metrics.beginCollect("", ["K"])
        Metrics.incCount("Line 1", "Metric 1", 5)
        Metrics.endCollect()

        Metrics.beginCollect("", ["K"])
        Metrics.endCollect()
        self.assertEqual(Metrics.dump(), {})

    def test_remove_use(self):
        """Test that removeUse works correctly"""
        Metrics.beginCollect("tmp/test_remove_use", ["K"])
        Metrics.traceRank("K")

        Metrics.addUse("K", 5)
        Metrics.incIter("K")
        Metrics.addUse("K", 7)
        Metrics.removeUse("K")
        Metrics.incIter("K")
        Metrics.addUse("K", 9)

        Metrics.endCollect()

        corr = [
            "K_pos,K\n",
            "0,5\n",
            "2,9\n"
        ]

        with open("tmp/test_remove_use-K.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_trace_rank_exists(self):
        """Test that the rank exists"""
        Metrics.beginCollect("", ["K"])

        with self.assertRaises(AssertionError):
            Metrics.traceRank("M")

        Metrics.endCollect()

