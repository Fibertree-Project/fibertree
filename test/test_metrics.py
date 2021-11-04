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
        self.assertFalse(Metrics.isCollecting())
        Metrics.beginCollect()
        self.assertTrue(Metrics.isCollecting())

    def test_end_collect(self):
        """Test that the endCollect() method ends collection"""
        Metrics.beginCollect()

        self.assertTrue(Metrics.isCollecting())
        Metrics.endCollect()
        self.assertFalse(Metrics.isCollecting())

    def test_empty_dump(self):
        """Test that if no metrics have been collected, the dump is empty"""
        Metrics.beginCollect()
        Metrics.endCollect()

        self.assertEqual(Metrics.dump(), {})

    def test_inc(self):
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

    def test_new_collection(self):
        """Test that a beginCollect() restarts collection"""
        Metrics.beginCollect()
        Metrics.incCount("Line 1", "Metric 1", 5)
        Metrics.endCollect()

        Metrics.beginCollect()
        self.assertEqual(Metrics.dump(), {})
