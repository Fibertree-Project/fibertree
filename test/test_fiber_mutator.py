import unittest
from fibertree.payload import Payload
from fibertree.fiber import Fiber

from fibertree.tensor_image import TensorImage

class TestFiberSplit(unittest.TestCase):

    def test_split_uniform_below(self):
        """Test splitUniform"""

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


# TBD: Add many more tests...


if __name__ == '__main__':
    unittest.main()

