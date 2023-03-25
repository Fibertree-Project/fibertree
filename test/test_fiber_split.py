import unittest
from fibertree import Payload
from fibertree import Fiber

from fibertree import TensorImage

class TestFiberSplit(unittest.TestCase):

    def test_split_uniform_empty(self):
        """Test splitUniform on empty fiber"""
        empty = Fiber()
        split = empty.splitUniform(5)

        # After we split, we need to make sure that we have actually added
        # another level to the empty fiber
        self.assertIsInstance(split.getDefault(), Fiber)

    def test_split_uniform(self):
        """Test splitUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 10, 30, 40 ]

        css = [ [ 0, 1, 9 ],
              [ 10, 12 ],
              [ 31 ],
              [ 41 ] ]

        pss = [ [ 1, 10, 20 ],
                [ 100, 120 ],
                [ 310 ],
                [ 410 ] ]

        ranges = [(0, 10), (10, 20), (30, 40), (40, 42)]

        split_ref_payloads = []
        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords)

        # Check the active range of the upper
        self.assertEqual(split.getActive(), (0, 42))

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())


    def test_split_uniform_on_int_coords_only(self):
        """Test that splitUnform works on integer coordinates only"""
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f = f.splitUniform(5)
        f = f.flattenRanks(style="tuple")

        with self.assertRaises(AssertionError):
            f.splitUniform((1, 2))

        with self.assertRaises(AssertionError):
            f.splitUniform(5)

    def test_split_uniform_then_flatten(self):
        """Test that flattenRanks() can undo splitUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords)

        #
        # Check that flattening after splitting gives us the same answer
        #
        self.assertEqual(split.flattenRanks(style="absolute"), f)


    def test_split_uniform_relative(self):
        """Test splitUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 10, 30, 40 ]

        css = [ [ 0, 1, 9 ],
              [ 0, 2 ],
              [ 1 ],
              [ 1 ] ]

        pss = [ [ 1, 10, 20 ],
                [ 100, 120 ],
                [ 310 ],
                [ 410 ] ]

        split_ref_payloads = []

        for (cs, ps) in zip(css, pss):
            split_ref_payloads.append(Fiber(cs, ps))

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords, relativeCoords=True)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])


    def test_split_uniform_relative_then_flatten(self):
        """Test that flattenRanks can undo splitUniform (relative)"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords, relativeCoords=True)

        #
        # Check the split
        #
        self.assertEqual(split.flattenRanks(style="relative"), f)

    def test_split_uniform_halo(self):
        """splitUniform with halo"""
        # Original Fiber
        c = [8, 9, 12, 15, 17, 32]
        p = [3, 4,  5,  6,  7,  8]
        f = Fiber(c, p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 8, 16, 24, 32]

        css = [ [ 8, 9 ],
              [ 8, 9, 12, 15, 17 ],
              [ 17 ],
              [ 32 ],
              [ 32 ] ]

        pss = [ [ 3, 4 ],
              [ 3, 4, 5, 6, 7 ],
              [ 7 ],
              [ 8 ],
              [ 8 ] ]

        ranges = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 33)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        coords = 8
        split = f.splitUniform(coords, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_uniform_halo_active_only(self):
        """splitUniform with halo, only active coordinates appear as non-halo elements"""
        # Original Fiber
        c = [0, 9, 12, 15, 17, 18]
        p = [3, 4,  5,  6,  7,  8]
        f = Fiber(c, p, active_range=(8, 16))

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [8, 12]

        css = [ [ 9, 12 ],
              [ 12, 15, 17] ]

        pss = [ [ 4, 5 ],
              [ 5, 6, 7 ] ]

        ranges = [(8, 12), (12, 16)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        coords = 4
        split = f.splitUniform(coords, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_uniform_halo_prev_active_halo(self):
        """splitUniform with halo, inactive coordinates can be haloed even if
        they should have been inside the active part of a partition"""
        # Original Fiber
        c = [0, 9, 12, 14, 17, 18, 20]
        p = [3, 4,  5,  6,  7,  8, 9]
        f = Fiber(c, p, active_range=(8, 16))

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [5, 10, 15]

        css = [ [ 9, 12 ],
                [ 12, 14, 17],
                [ 17, 18 ] ]

        pss = [ [ 4, 5 ],
              [ 5, 6, 7 ],
              [ 7, 8 ] ]

        ranges = [(8, 10), (10, 15), (15, 16)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        coords = 5
        split = f.splitUniform(coords, halo=3)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_uniform_build_halo_once(self):
        """splitUniform, make sure that if the halo for the last partition has already
        been built, we do not try to build it again"""
        c = [0, 9, 12, 14, 17, 19, 20]
        p = [3, 4,  5,  6,  7,  8, 9]
        f = Fiber(c, p, active_range=(8, 16))

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 10]

        css = [ [ 9, 12 ],
                [ 12, 14, 17] ]

        pss = [ [ 4, 5 ],
              [ 5, 6, 7 ] ]

        ranges = [(8, 10), (10, 16)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords, halo=3)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_uniform_halo_not_bigger_than_step(self):
        """splitUniform, halo cannot be bigger than the step"""
        # Original Fiber
        c = [8, 9, 12, 15, 17, 32]
        p = [3, 4,  5,  6,  7,  8]
        f = Fiber(c, p)

        with self.assertRaises(AssertionError):
            f.splitUniform(3, halo=5)

    def test_split_nonuniform_empty(self):
        """Test splitNonUniform on empty fiber"""
        empty = Fiber()
        split = empty.splitNonUniform([1, 5, 17])

        # After we split, we need to make sure that we have actually added
        # another level to the empty fiber
        self.assertIsInstance(split.getDefault(), Fiber)

    def test_split_nonuniform_empty_split(self):
        """Test that splitNonUniform works when the split list has no elements"""
        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        splits = []
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(split, Fiber())

    def test_split_nonuniform_one_split(self):
        """Test that splitNonUniform works when the split list has one element"""
        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 31, 41 ] ]

        pss = [ [ 310, 410 ] ]

        ranges = [(20, 42)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [20]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_nonuniform_all_before(self):
        """Test splitNonUniform - all coordinates are before the first split"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        splits = [50, 60, 70]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(split, Fiber())

    def test_split_nonuniform1(self):
        """Test splitNonUniform - starting at coordinate 0"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 0, 1, 9, 10 ],
              [ 12 ],
              [ 31, 41 ] ]

        pss = [ [ 1, 10, 20, 100 ],
                [ 120 ],
                [ 310, 410 ] ]

        ranges = [(0, 12), (12, 31), (31, 42)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [0, 12, 31]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_nonuniform2(self):
        """Test splitNonUniform - not starting at coordinate 0"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10 ],
              [ 12 ],
              [ 31, 41 ] ]

        pss = [ [ 20, 100 ],
                [ 120 ],
                [ 310, 410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        #
        # Do the split
        #
        splits = [8, 12, 31]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])

    def test_split_nonuniform_fiber(self):
        """Test splitNonUniform - where the input is a fiber"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 0, 1, 9, 10 ],
              [ 12 ],
              [ 31, 41 ] ]

        pss = [ [ 1, 10, 20, 100 ],
                [ 120 ],
                [ 310, 410 ] ]

        ranges = [(0, 12), (12, 31), (31, 42)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = Fiber([0, 12, 31], [1, 1, 1])
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits.coords[i])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_nonuniform_active_only(self):
        """Test splitNonUniform only iterates over the active_range of the fiber"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p, active_range=(8, 20))

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10 ],
              [ 12 ] ]

        pss = [ [ 20, 100 ],
                [ 120 ] ]

        ranges = [(8, 12), (12, 20)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [0, 12, 31]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_non_uniform_int_splits(self):
        """Test splitNonUniform only works with integer splits"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 13, 31, 41]
        p = [ 1, 10, 20, 100, 120, 130, 310, 410 ]

        f = Fiber(c,p)

        splits = [8, (19, 31), 40]

        with self.assertRaises(AssertionError):
            f.splitNonUniform(splits)

    def test_split_nonuniform_halo(self):
        """Test splitNonUniform with a halo"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 13, 31, 41]
        p = [ 1, 10, 20, 100, 120, 130, 310, 410 ]

        f = Fiber(c,p, shape=50)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10, 12],
              [ 12, 13 ],
              [ 31 ],
              [ 31, 41 ] ]

        pss = [ [ 20, 100, 120 ],
                [ 120, 130 ],
                [ 310 ],
                [ 310, 410 ] ]

        ranges = [(8, 11), (11, 15), (20, 30), (30, 50)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [8, 11, 15, 20, 30]
        split = f.splitNonUniform(splits, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, ranges[i][0])
            self.assertEqual(sp, split_ref[i])

    def test_split_nonuniform_halo_outside_active(self):
        """Test splitNonUniform with a halo, allowing the halo to extend
        outside the active_range"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 13, 31, 41]
        p = [ 1, 10, 20, 100, 120, 130, 310, 410 ]

        f = Fiber(c,p, active_range=(8, 31))

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10, 12],
              [ 12, 13 ],
              [ 31 ],
              [ 31 ] ]

        pss = [ [ 20, 100, 120 ],
                [ 120, 130 ],
                [ 310 ],
                [ 310 ] ]

        split_ref = []

        ranges = [(8, 11), (11, 20), (20, 30), (30, 31)]
        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [2, 8, 11, 20, 30, 40]
        split = f.splitNonUniform(splits, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(split_ref))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, ranges[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_nonuniform_halo_outside_active2(self):
        """Test splitNonUniform with a halo, making normally active coordinates
        halo coordinates"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 13, 31, 34, 41]
        p = [ 1, 10, 20, 100, 120, 130, 310, 340, 410 ]

        f = Fiber(c,p, active_range=(8, 30))

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10, 12],
              [ 12, 13 ],
              [ 31 ] ]

        pss = [ [ 20, 100, 120 ],
                [ 120, 130 ],
                [ 310 ] ]

        ranges = [(8, 11), (11, 20), (20, 30)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [8, 11, 20, 40]
        split = f.splitNonUniform(splits, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(split_ref))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_nonuniform_halo_outside_active3(self):
        """Test splitNonUniform with a halo, ensure that the halo is only
        built once"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 13, 31, 41]
        p = [ 1, 10, 20, 100, 120, 130, 310, 410 ]

        f = Fiber(c,p, shape=40)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10, 12],
              [ 12, 13 ],
              [ 31 ],
              [ 31, 41 ] ]

        pss = [ [ 20, 100, 120 ],
                [ 120, 130 ],
                [ 310 ],
                [ 310, 410 ] ]

        ranges = [(8, 11), (11, 15), (20, 30), (30, 40)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        splits = [8, 11, 15, 20, 30, 50]
        split = f.splitNonUniform(splits, halo=2)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, ranges[i][0])
            self.assertEqual(sp, split_ref[i])

    def test_split_nonuniform_valid_halo(self):
        """Test that splitNonUniform only accepts valid halos"""
        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        splits = [8, 11, 30]

        with self.assertRaises(AssertionError):
            f.splitNonUniform(splits, halo=(10, 29))

        with self.assertRaises(AssertionError):
            f.splitNonUniform(splits, halo=100)

    def test_split_nonuniform_then_flatten(self):
        """Test that flattenRanks can undo splitNonUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        splits = [0, 12, 31]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        self.assertEqual(split.flattenRanks(style="absolute"), f)

    def test_split_nonuniform_relative(self):
        """Test that relative coordinates on splitNonUniform"""
        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        splits = [0, 12, 31]
        split = f.splitNonUniform(splits, relativeCoords=True)

        css = [[0, 1, 9, 10], [0], [0, 10]]
        pss = [[1, 10, 20, 100], [120], [310, 410]]

        ans = Fiber()
        self.assertEqual(len(split), len(css))
        for c, cs, ps in zip(splits, css, pss):
            part = Fiber(cs, ps)
            ans.append(c, part)

        self.assertEqual(split, ans)

    def test_split_equal_empty(self):
        """Test splitEqual on empty fiber"""
        empty = Fiber()
        split = empty.splitEqual(3)

        # After we split, we need to make sure that we have actually added
        # another level to the empty fiber
        self.assertIsInstance(split.getDefault(), Fiber)


    def test_split_equal(self):
        """Test splitEqual"""

        #
        # Create the fiber to be split
        #
        c = [1, 2, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [1, 2 ],
                [9, 10 ],
                [12, 31 ],
                [41 ] ]

        pss = [ [1, 10 ],
                [20, 100 ],
                [120, 310 ],
                [410 ] ]

        ranges = [(0, 9), (9, 12), (12, 41), (41, 42)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_equal_active_only(self):
        """Test splitEqual"""

        #
        # Create the fiber to be split
        #
        c = [1, 2, 9, 10, 12, 30, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c, p, active_range=(8, 32))

        #
        # Create list of reference fibers after the split
        #
        css = [ [9, 10 ],
                [12, 30 ] ]

        pss = [ [20, 100 ],
                [120, 310 ] ]

        ranges = [(8, 12), (12, 32)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_equal_relative(self):
        """Test splitEqual with relative coordinates"""
        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [0, 1 ],
                [0, 1 ],
                [0, 19 ],
                [0 ] ]

        pss = [ [1, 10 ],
                [20, 100 ],
                [120, 310 ],
                [410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        coords = [0, 9, 12, 41]
        ans = Fiber(coords, split_ref)

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size, relativeCoords=True)

        #
        # Check the split
        #
        self.assertEqual(split, ans)

    def test_split_equal_int_step_halo_only(self):
        """Test that splitEqual only works on an integer step, integer halo, and no halo with tuple coordinates"""
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f = f.splitUniform(5)
        f = f.flattenRanks(style="tuple")

        with self.assertRaises(AssertionError):
            f.splitEqual((1, 2))

        with self.assertRaises(AssertionError):
            f.splitEqual(5, halo=(1, 2))

        with self.assertRaises(AssertionError):
            f.splitEqual(5, halo=3)

    def test_split_equal_tuple_coords(self):
        """Test that splitEqual works with tuple coordinates"""
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f = f.splitUniform(5)
        f = f.flattenRanks(style="tuple")

        split = f.splitEqual(3)

        corr = Fiber([(0, 0), (10, 10), (40, 41)], [Fiber([(0, 0), (0, 1), (5, 9)], [1, 10, 20]), Fiber([(10, 10), (10, 12), (30, 31)], [100, 120, 310]), Fiber([(40, 41)], [410])])

        self.assertEqual(split, corr)

    def test_split_equal_halo(self):
        """splitEqual with halo"""
        # Original Fiber
        c = [0, 1, 8, 9, 12, 15, 17, 19]
        p = [1, 2, 3, 4,  5,  6,  7,  8]
        f = Fiber(c, p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 9, 17]

        css = [ [ 0, 1, 8, 9 ],
              [ 9, 12, 15, 17, 19 ],
              [ 17, 19 ] ]

        pss = [ [ 1, 2, 3, 4 ],
              [ 4, 5, 6, 7, 8 ],
              [ 7, 8 ] ]

        ranges = [(0, 9), (9, 17), (17, 20)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        size = 3
        split = f.splitEqual(size, halo=3)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_equal_halo_active_only(self):
        """splitEqual with halo"""
        # Original Fiber
        c = [0, 8, 9, 11, 12, 15, 17, 18, 25]
        p = [1, 2, 3, 4,  5,  6,  7,  8,  9 ]
        f = Fiber(c, p, active_range=(8, 16))

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [8, 12]

        css = [ [ 8, 9, 11, 12],
              [ 12, 15, 17, 18] ]

        pss = [ [ 2, 3, 4, 5 ],
              [ 5, 6, 7, 8 ] ]

        ranges = [(8, 12), (12, 16)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        size = 3
        split = f.splitEqual(size, halo=3)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_equal_correct_final_halo(self):
        """splitEqual - Correctly build the final halo"""
        c = list(range(40))
        p = list(range(40))
        p[0] = 100

        f = Fiber(c, p, active_range=(0, 36))

        split_ref_coords = [0, 5, 10, 15, 20, 25, 30, 35]

        css = [ [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9, 10, 11, 12, 13],
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                [15, 16, 17, 18, 19, 20, 21, 22, 23],
                [20, 21, 22, 23, 24, 25, 26, 27, 28],
                [25, 26, 27, 28, 29, 30, 31, 32, 33],
                [30, 31, 32, 33, 34, 35, 36, 37, 38],
                [35, 36, 37, 38, 39] ]

        pss = [ [100, 1, 2, 3, 4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9, 10, 11, 12, 13],
                [10, 11, 12, 13, 14, 15, 16, 17, 18],
                [15, 16, 17, 18, 19, 20, 21, 22, 23],
                [20, 21, 22, 23, 24, 25, 26, 27, 28],
                [25, 26, 27, 28, 29, 30, 31, 32, 33],
                [30, 31, 32, 33, 34, 35, 36, 37, 38],
                [35, 36, 37, 38, 39] ]

        ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 36)]

        split_ref_payloads = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref_payloads.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        size = 5
        split = f.splitEqual(size, halo=4)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])
            self.assertEqual(sp.getActive(), split_ref_payloads[i].getActive())

    def test_split_equal_halo_not_bigger_than_step(self):
        """splitEqual, halo cannot be bigger than the step"""
        # Original Fiber
        c = [8, 9, 12, 15, 17, 32]
        p = [3, 4,  5,  6,  7,  8]
        f = Fiber(c, p)

        with self.assertRaises(AssertionError):
            f.splitEqual(3, halo=5)

    def test_split_equal_then_flatten(self):
        """Test that flattenRanks can undo splitEqual"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size)

        #
        # Check the split
        #
        self.assertEqual(split.flattenRanks(style="absolute"), f)


    def test_split_unequal_empty(self):
        """Test splitUnEqual on empty fiber"""
        empty = Fiber()
        split = empty.splitUnEqual([1, 5, 17])

        # After we split, we need to make sure that we have actually added
        # another level to the empty fiber
        self.assertIsInstance(split.getDefault(), Fiber)


    def test_split_unequal(self):
        """Test splitUnequal"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [0],
                [1, 9],
                [10, 12, 31, 41] ]

        pss = [ [1],
                [10, 20],
                [ 100, 120, 310, 410 ] ]

        ranges = [(0, 1), (1, 10), (10, 42)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        sizes = [1, 2, 4]
        split = f.splitUnEqual(sizes)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())


    def test_split_unequal_active_only(self):
        """Test splitUnequal - count only the coordinates in the active range"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p, active_range=(8, 32))

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 9, 10, 12 ],
                [ 31 ] ]

        pss = [ [ 20, 100, 120],
                [ 310 ] ]

        ranges = [(8, 31), (31, 32)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        sizes = [3, 2]
        split = f.splitUnEqual(sizes)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())


    def test_split_unequal_relative(self):
        """Test splitUnequal on relative coordinates"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [0],
                [0, 8],
                [0, 2, 21, 31] ]

        pss = [ [1],
                [10, 20],
                [ 100, 120, 310, 410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        coords = [0, 1, 10]
        ans = Fiber(coords, split_ref)

        #
        # Do the split
        #
        sizes = [1, 2, 4]
        split = f.splitUnEqual(sizes, relativeCoords=True)

        #
        # Check the split
        #
        self.assertEqual(split, ans)


    def test_split_unequal_halo(self):
        """Test splitUnequal - with halo (all parts are filled)"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41, 51]
        p = [1, 10, 20, 100, 120, 310, 410, 510 ]

        f = Fiber(c,p, active_range=(0, 41))

        #
        # Create list of reference fibers after the split
        #
        css = [ [0, 1],
                [1, 9, 10],
                [10, 12, 31, 41] ]

        pss = [ [1, 10],
                [10, 20, 100],
                [ 100, 120, 310, 410 ] ]

        ranges = [(0, 1), (1, 10), (10, 41)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        sizes = [1, 2, 3]
        split = f.splitUnEqual(sizes, halo=1)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_unequal_halo2(self):
        """Test splitUnequal - With halo (last part does not finish) """

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41, 51]
        p = [1, 10, 20, 100, 120, 310, 410, 510 ]

        f = Fiber(c,p, active_range=(0, 41))

        #
        # Create list of reference fibers after the split
        #
        css = [ [0, 1],
                [1, 9, 10],
                [10, 12, 31, 41] ]

        pss = [ [1, 10],
                [10, 20, 100],
                [ 100, 120, 310, 410 ] ]

        ranges = [(0, 1), (1, 10), (10, 41)]

        split_ref = []

        for cs, ps, range_ in zip(css, pss, ranges):
            split_ref.append(Fiber(cs, ps, active_range=range_))

        #
        # Do the split
        #
        sizes = [1, 2, 5, 16]
        split = f.splitUnEqual(sizes, halo=1)

        #
        # Check the split
        #
        self.assertEqual(len(split), len(css))
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])
            self.assertEqual(sp.getActive(), split_ref[i].getActive())

    def test_split_unequal_asserts(self):
        """Test the error-checking for splitUnEqual"""
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f_split = f.splitUniform(5)
        f_flat = f_split.flattenRanks(style="tuple")

        with self.assertRaises(AssertionError):
            f.splitUnEqual([3, 4, 5], halo=(2, 3))

        with self.assertRaises(AssertionError):
            f.splitUnEqual([3, 4, (5, 6)])

        with self.assertRaises(AssertionError):
            f_flat.splitUnEqual([3, 4, 5], halo=(2, 3))

        with self.assertRaises(AssertionError):
            f_split.splitUnEqual([3, 4, (5, 6)])

        with self.assertRaises(AssertionError):
            f.splitUnEqual([3, 4, 5], halo=4)

        with self.assertRaises(AssertionError):
            f_flat.splitUnEqual([3, 4, 5], halo=2)

    def test_split_unequal_then_flatten(self):
        """Test that flattenRanks can undo splitUnequal"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 1, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Do the split
        #
        sizes = [1, 2, 4]
        split = f.splitUnEqual(sizes)

        #
        # Check the split
        #
        self.assertEqual(split.flattenRanks(style="absolute"), f)

    @staticmethod
    def _make_fiber_a():

        f = Fiber([0, 1, 2, 10, 12, 31, 41], [ 0, 10, 20, 100, 120, 310, 410 ])
        return f


if __name__ == '__main__':
    unittest.main()

