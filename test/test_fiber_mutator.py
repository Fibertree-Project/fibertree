import unittest

from copy import deepcopy

from fibertree import Payload
from fibertree import Fiber

from fibertree import TensorImage

class TestFiberMutator(unittest.TestCase):

    def test_swapRanks_empty(self):
        """Test that swapRanks raises an error if the fiber is empty"""
        z_m = Fiber()

        with self.assertRaises(AssertionError):
            z_m.swapRanks()

    def test_swapRanks_eager_only(self):
        """Test that swapRanks works in eager mode only"""
        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.swapRanks()

    def test_split_uniform_below(self):
        """Test splitUniformBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        f.splitUniformBelow(10, depth=0)

        f0_split = f0.splitUniform(10)
        f1_split = f1.splitUniform(10)

        f_ref = Fiber(c, [f0_split, f1_split])

        self.assertEqual(f, f_ref)

    def test_split_nonuniform_below(self):
        """Test splitNonUniformBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        f.splitNonUniformBelow([0, 20,45, 50], depth=0)

        f0_split = f0.splitNonUniform([0, 20, 45, 50])
        f1_split = f1.splitNonUniform([0, 20, 45, 50])

        f_ref = Fiber(c, [f0_split, f1_split])

        self.assertEqual(f, f_ref)


    def test_split_equal_below(self):
        """Test splitEqualBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        f.splitEqualBelow(4, depth=0)

        f0_split = f0.splitEqual(4)
        f1_split = f1.splitEqual(4)

        f_ref = Fiber(c, [f0_split, f1_split])

        self.assertEqual(f, f_ref)


    def test_split_unequal_below(self):
        """Test splitUnEqualBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        f.splitUnEqualBelow([3, 3, 1], depth=0)

        f0_split = f0.splitUnEqual([3, 3, 1])
        f1_split = f1.splitUnEqual([3, 3, 1])

        f_ref = Fiber(c, [f0_split, f1_split])

        self.assertEqual(f, f_ref)


    def test_flatten_below(self):
        """Test {,un}flattenRanksBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        # This just creates another level...
        f.splitUnEqualBelow([3, 3, 1], depth=0)
        f_ref = deepcopy(f)

        # Flattening and unflattening should do nothing
        f.flattenRanksBelow()
        f.unflattenRanksBelow()

        self.assertEqual(f, f_ref)


    def test_swap_below(self):
        """Test swapRanksBelow"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        # This just creates another level...
        f.splitUnEqualBelow([3, 3, 1], depth=0)
        f_ref = deepcopy(f)

        # Swapping twice should do nothing
        f.swapRanksBelow()
        f.swapRanksBelow()

        self.assertEqual(f, f_ref)

    def test_split_equal_below_deep(self):
        """Test splitEqualBelow with depth=1"""

        c0 = [0, 1, 9, 10, 12, 31, 41]
        p0 = [ 0, 10, 20, 100, 120, 310, 410 ]
        f0 = Fiber(c0, p0)

        c1 = [1, 2, 10, 11, 13, 32, 42]
        p1 = [ 1, 11, 21, 101, 121, 311, 411 ]
        f1 = Fiber(c1, p1)

        c = [2, 4]
        f = Fiber(c, [f0, f1])

        # This just creates another level...
        f.splitUnEqualBelow([3, 3, 1], depth=0)
        f_ref = deepcopy(f)

        f.splitEqualBelow(2, depth=1)

        for fc, fp in f_ref:
            fp.splitEqualBelow(2)

        self.assertEqual(f, f_ref)


if __name__ == '__main__':
    unittest.main()

