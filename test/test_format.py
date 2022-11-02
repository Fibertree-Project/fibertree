"""Tests of the tensor Format"""

import unittest
import yaml

from fibertree import Tensor
from fibertree.model import Format

class TestFormat(unittest.TestCase):
    """Tests of the tensor Format"""

    def setUp(self):
        self.t = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        spec = """
        root:
            hbits: 128
            pbits: 32
        M:
            format: U
            fhbits: 32
            pbits: 32
        K:
            format: C
            rhbits: 128
            cbits: 32
            pbits: 64
            layout: interleaved
        """
        self.spec = yaml.safe_load(spec)

    def test_bad_format(self):
        """Test if the format is bad"""
        spec = """
        M:
            format: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_layout(self):
        """Test if the layout is bad"""
        spec = """
        M:
            layout: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_rhbits(self):
        """Test if the rhbits is bad"""
        spec = """
        M:
            rhbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_fhbits(self):
        """Test if the fhbits is bad"""
        spec = """
        M:
            fhbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_cbits(self):
        """Test if the cbits is bad"""
        spec = """
        M:
            cbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_pbits(self):
        """Test if the pbits is bad"""
        spec = """
        M:
            pbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))


    def test_bad_field(self):
        """Test if there is an extra field"""
        spec = """
        M:
            extra: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_hbits_root(self):
        """Test if the hbits is bad on the root"""
        spec = """
        root:
            hbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_pbits_root(self):
        """Test if the pbits is bad on the root"""
        spec = """
        root:
            pbits: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))

    def test_bad_field_root(self):
        """Test if there is an extra field on root"""
        spec = """
        root:
            extra: Y
        """

        with self.assertRaises(AssertionError):
            f = Format(self.t, yaml.safe_load(spec))


    def test_bad_fiber_footprint(self):
        f = Format(self.t, {})

        with self.assertRaises(AssertionError):
            f.getFiber(1, 0)

    def test_get_cbits(self):
        """Test getCBits"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getCBits("M"), 0)
        self.assertEqual(f.getCBits("K"), 32)

    def test_get_fhbits(self):
        """Test getFHBits"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getFHBits("M"), 32)
        self.assertEqual(f.getFHBits("K"), 0)

    def test_get_pbits(self):
        """Test getPBits"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getPBits("M"), 32)
        self.assertEqual(f.getPBits("K"), 64)

    def test_get_rhbits(self):
        """Test getRHBits"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getRHBits("M"), 0)
        self.assertEqual(f.getRHBits("K"), 128)

    def test_get_elem(self):
        """Test getElem"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getElem("K", "coord"), 32)
        self.assertEqual(f.getElem("K", "payload"), 64)
        self.assertEqual(f.getElem("K", "elem"), 96)

        with self.assertRaises(AssertionError):
            f.getElem("K", "foo")

    def test_get_layout(self):
        """Test getLayout"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getLayout("M"), "contiguous")
        self.assertEqual(f.getLayout("K"), "interleaved")

    def test_fiber_footprint(self):
        """Test the fiber footprint"""
        f = Format(self.t, self.spec)

        self.assertEqual(f.getFiber(), 32 + 7 * 32)
        self.assertEqual(f.getFiber(1), 3 * 32 + 3 * 64)

    def test_rank_footprint(self):
        """Test the rank footprint"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getRank("K"), 128 + (3 + 2 + 2 + 2) * (32 + 64))

    def test_root_footprint(self):
        """Test the root footprint"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getRoot(), 128 + 32)

    def test_subtree_footprint(self):
        """Test the subtree footprint"""
        f = Format(self.t, self.spec)

        self.assertEqual(f.getSubTree(), 32 + 7 * 32 + (3 + 2 + 2 + 2) * (32 + 64))
        self.assertEqual(f.getSubTree(1), 3 * 32 + 3 * 64)
        self.assertEqual(f.getSubTree(1, 2), 32 + 64)

    def test_subtree_footprint_dense(self):
        """Test the subtree footprint for dense tensors"""
        spec = """
        M:
            format: U
        K:
            format: U
            pbits: 32
        """
        tensor = Tensor(rank_ids=["M", "K"], shape=[5, 7])
        f = Format(tensor, yaml.safe_load(spec))
        self.assertEqual(f.getSubTree(), 5 * 7 * 32)

    def test_tensor_footprint(self):
        """Test the tensor footprint"""
        f = Format(self.t, self.spec)
        self.assertEqual(f.getTensor(),
            128 + 32 + 32 + 7 * 32 + 128 + (3 + 2 + 2 + 2) * (32 + 64))
