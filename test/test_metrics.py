import unittest

from fibertree import *

class TestMetrics(unittest.TestCase):
    def test_is_collecting(self):
        """Test the isCollecting() method directly"""
        Metrics.collecting = True
        self.assertTrue(Metrics.isCollecting())

        Metrics.collecting = False
        self.assertFalse(Metrics.isCollecting())

    def test_begin_collect(self):
        """Test that the beginCollect() method begins collection"""
        # NDN: This is a hack and should be fixed
        Metrics.collecting = False

        self.assertFalse(Metrics.isCollecting())
        Metrics.beginCollect(["K"])
        self.assertTrue(Metrics.isCollecting())

        Metrics.endCollect()

    def test_end_collect(self):
        """Test that the endCollect() method ends collection"""
        Metrics.beginCollect(["K"])

        self.assertTrue(Metrics.isCollecting())
        Metrics.endCollect()
        self.assertFalse(Metrics.isCollecting())

    def test_empty_dump(self):
        """Test that if no metrics have been collected, the dump is empty"""
        Metrics.beginCollect(["K"])
        Metrics.endCollect()

        self.assertEqual(Metrics.dump(), {})

    def test_inc_count(self):
        """Test that inc updates the correct line/metric and adds new entries
        to the metrics dictionary if the line/metric do not already exist"""
        Metrics.beginCollect(["K"])
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
        Metrics.beginCollect(["N", "M"])
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
        Metrics.beginCollect(["N", "M"])

        Metrics.clrIter("M")
        Metrics.incIter("N")

        self.assertEqual(Metrics.getIter(), (1, 0))

        Metrics.endCollect()

    def test_new_collection(self):
        """Test that a beginCollect() restarts collection"""
        Metrics.beginCollect(["K"])
        Metrics.incCount("Line 1", "Metric 1", 5)
        Metrics.endCollect()

        Metrics.beginCollect(["K"])
        Metrics.endCollect()
        self.assertEqual(Metrics.dump(), {})
