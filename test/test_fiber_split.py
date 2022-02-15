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
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 10, 30, 40 ]

        css = [ [ 0, 1, 9 ],
              [ 10, 12 ],
              [ 31 ],
              [ 41 ] ]

        pss = [ [ 0, 10, 20 ],
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
        split = f.splitUniform(coords)

        #
        # Check the split
        #
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])


    def test_split_uniform_then_flatten(self):
        """Test that flattenRanks() can undo splitUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        split_ref_coords = [0, 10, 30, 40 ]

        css = [ [ 0, 1, 9 ],
              [ 0, 2 ],
              [ 1 ],
              [ 1 ] ]

        pss = [ [ 0, 10, 20 ],
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
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, split_ref_coords[i])
            self.assertEqual(sp, split_ref_payloads[i])


    def test_split_uniform_relative_then_flatten(self):
        """Test that flattenRanks can undo splitUniform (relative)"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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

    def test_split_nonuniform_empty(self):
        """Test splitNonUniform on empty fiber"""
        empty = Fiber()
        split = empty.splitNonUniform([1, 5, 17])

        # After we split, we need to make sure that we have actually added
        # another level to the empty fiber
        self.assertIsInstance(split.getDefault(), Fiber)

    def test_split_nonuniform1(self):
        """Test splitNonUniform - starting at coordinate 0"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [ 0, 1, 9, 10 ],
              [ 12 ],
              [ 31, 41 ] ]

        pss = [ [ 0, 10, 20, 100 ],
                [ 120 ],
                [ 310, 410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        #
        # Do the split
        #
        splits = [0, 12, 31]
        split = f.splitNonUniform(splits)

        #
        # Check the split
        #
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])

    def test_split_nonuniform2(self):
        """Test splitNonUniform - not starting at coordinate 0"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, splits[i])
            self.assertEqual(sp, split_ref[i])

    def test_split_nonuniform_then_flatten(self):
        """Test that flattenRanks can undo splitNonUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [0, 1 ],
                [9, 10 ],
                [12, 31 ],
                [41 ] ]

        pss = [ [0, 10 ],
                [20, 100 ],
                [120, 310 ],
                [410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size)

        #
        # Check the split
        #
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])

    def test_split_equal_then_flatten(self):
        """Test that flattenRanks can undo splitEqual"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        #
        # Do the split
        #
        sizes = [1, 2, 4]
        split = f.splitUnEqual(sizes)

        #
        # Check the split
        #
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, css[i][0])
            self.assertEqual(sp, split_ref[i])


    def test_split_unequal_then_flatten(self):
        """Test that flattenRanks can undo splitUnequal"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

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


    def test_split_equal_partioned(self):
        """Test splitEqual(2, partitions=2)"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        a_coords = [0, 12]
        a1 = Fiber([0, 1], [0, 10])
        a2 = Fiber([12, 31], [120, 310])
        a = Fiber(coords=a_coords, payloads=[a1, a2])

        b_coords = [9, 41]
        b1 = Fiber([9, 10], [20, 100])
        b2 = Fiber([41], [410])
        b = Fiber(coords=b_coords, payloads=[b1, b2])

        split_ref = Fiber(payloads=[a, b])

        #
        # Do the split
        #
        size = 2
        split = f.splitEqual(size, partitions=2)

        #
        # Check the split
        #
        self.assertEqual(split, split_ref)

    @staticmethod
    def _make_fiber_a():

        f = Fiber([0, 1, 2, 10, 12, 31, 41], [ 0, 10, 20, 100, 120, 310, 410 ])
        return f


if __name__ == '__main__':
    unittest.main()

