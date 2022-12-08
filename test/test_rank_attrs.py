import unittest

from fibertree import *
from fibertree.core.rank_attrs import RankAttrs

class TestRankAttrs(unittest.TestCase):
    def test_id(self):
        """Test get and set rank id"""
        attrs = RankAttrs("K")
        self.assertEqual(attrs.getId(), "K")

        attrs.setId("M")
        self.assertEqual(attrs.getId(), "M")

    def test_default(self):
        """Test get and set default"""
        attrs = RankAttrs("K")
        with self.assertRaises(AssertionError):
            attrs.getDefault()

        attrs.setDefault(5)
        self.assertEqual(attrs.getDefault(), 5)

    def test_format(self):
        """Test get and set format"""
        attrs = RankAttrs("K")
        self.assertEqual(attrs.getFormat(), "C")

        attrs.setFormat("U")
        self.assertEqual(attrs.getFormat(), "U")

        with self.assertRaises(AssertionError):
            attrs.setFormat("K")

    def test_shape(self):
        """Test get and set shape"""
        attrs = RankAttrs("K", 10)
        self.assertEqual(attrs.getShape(), 10)

        attrs.setShape(100)
        self.assertEqual(attrs.getShape(), 100)

    def test_estim_shape(self):
        """Test that the estimated shape is set correctly"""
        attrs = RankAttrs("K")
        self.assertTrue(attrs.getEstimatedShape())

        attrs.setEstimatedShape(False)
        self.assertFalse(attrs.getEstimatedShape())

        attrs = RankAttrs("K", 10)
        self.assertFalse(attrs.getEstimatedShape())

    def test_eq(self):
        """Test =="""
        attrs1 = RankAttrs("K")
        attrs1.setFormat("U").setDefault(3).setId("M").setShape(20)

        attrs2 = RankAttrs("K")
        attrs2.setFormat("U").setDefault(3).setId("M").setShape(20)

        attrs3 = RankAttrs("K")

        self.assertEqual(attrs1, attrs2)
        self.assertNotEqual(attrs1, attrs3)
        self.assertNotEqual(attrs1, "foo")

    def test_repr(self):
        """Test repr"""
        attrs = RankAttrs("K")
        attrs.setFormat("U").setDefault(3).setId("M").setShape(20)

        self.assertEqual(repr(attrs), "(RankAttrs, 'M', True, 20, 'U', Payload(3))")
