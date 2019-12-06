import unittest
from fibertree.payload import Payload
from fibertree.fiber import Fiber


class TestFiberSplit(unittest.TestCase):

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
        css = [ [ 0, 1, 9 ],
              [ 10, 12 ],
              [ 31 ],
              [ 41 ] ]

        pss = [ [ 0, 10, 20 ],
                [ 100, 120 ],
                [ 310 ],
                [ 410 ] ]

        split_ref = []

        for (cs, ps) in zip(css, pss):
            split_ref.append(Fiber(cs, ps))

        #
        # Do the split
        #
        coords = 10
        split = f.splitUniform(coords)

        #
        # Check the split
        #
        for i, (sc, sp)  in enumerate(split):
            self.assertEqual(sc, i)
            self.assertEqual(sp, split_ref[i])
        

    def test_split_nonuniform(self):
        """Test splitNonUniform"""

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
            self.assertEqual(sc, i)
            self.assertEqual(sp, split_ref[i])
        
    def test_split_equal(self):
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
            self.assertEqual(sc, i)
            self.assertEqual(sp, split_ref[i])
        
    def test_split_unequal(self):
        """Test splitNonUniform"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]
        
        f = Fiber(c,p)

        #
        # Create list of reference fibers after the split
        #
        css = [ [0],
                [1, 9],
                [10, 12, 31, 41] ]

        pss = [ [0],
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
            self.assertEqual(sc, i)
            self.assertEqual(sp, split_ref[i])
        

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
        a1 = Fiber([0, 1], [0, 10])
        a2 = Fiber([12, 31], [120, 310])
        a = Fiber(payloads=[a1, a2])

        b1 = Fiber([9, 10], [20, 100])
        b2 = Fiber([41], [410])
        b = Fiber(payloads=[b1, b2])

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

