import unittest

from fibertree import Payload
from fibertree import Fiber

class TestFiberPrint(unittest.TestCase):

    def test_print_1D(self):
        """Test str format 1D"""

        c = [2, 4, 6, 8]
        p = [3, 5, 7, 9]

        a = Fiber(c, p)

        ss = f"{a:n*}"
        ss_ref = "F/[(2 -> <3>) \n   (4 -> <5>) \n   (6 -> <7>) \n   (8 -> <9>) ]"

        self.assertEqual(ss, ss_ref)

        sr = f"{a!r}"
        sr_ref = "Fiber([2, 4, 6, 8], [3, 5, 7, 9])"

        self.assertEqual(sr, sr_ref)


    def test_print_2D_flattened(self):
        """Test str format 2D flattened"""

        c = [(2,3), (2,4), (3,1), (8,2)]
        p = [3, 5, 7, 9]

        a = Fiber(c, p)

        ss = f"{a:n*}"
        ss_ref = "F/[((2, 3) -> <3>) \n   ((2, 4) -> <5>) \n   ((3, 1) -> <7>) \n   ((8, 2) -> <9>) ]"

        self.assertEqual(ss, ss_ref)

        sr = f"{a!r}"
        sr_ref = "Fiber([(2, 3), (2, 4), (3, 1), (8, 2)], [3, 5, 7, 9])"
        self.assertEqual(sr, sr_ref)


    def test_print_2D(self):
        """Test str format 2D"""

        c0 = [2, 4, 6, 8]
        p0 = [3, 5, 7, 9]
        f0 = Fiber(c0, p0)

        c1 = [3, 5, 7]
        p1 = [4, 6, 8]
        f1 = Fiber(c1, p1)

        c = [2,5]

        a = Fiber(c, [f0, f1])

        # Plain formating

        s = f"{a}"

        s_ref = "F/[( 2 -> F/[(2 -> <3>) " + \
                 "(4 -> <5>) "             + \
                 " ... "                   + \
                 " ... ])"                 + \
                 "( 5 -> F/[(3 -> <4>) "   + \
                 "(5 -> <6>) "             + \
                 " ... "                   + \
                 " ... ])"

        self.assertEqual(s, s_ref)

        # Plain formating, no cutoff

        ss = f"{a:*}"

        ss_ref = "F/[( 2 -> F/[(2 -> <3>) " + \
                 "(4 -> <5>) "              + \
                 "(6 -> <7>) "              + \
                 "(8 -> <9>) ])"            + \
                 "( 5 -> F/[(3 -> <4>) "    + \
                 "(5 -> <6>) "              + \
                 "(7 -> <8>) ])"

        self.assertEqual(ss, ss_ref)


        # Format with newline and cutoff

        sn = f"{a:n}"

        sn_ref = "F/[( 2 -> F/[(2 -> <3>) \n" + \
                 "             (4 -> <5>) \n" + \
                 "              ... \n"       + \
                 "              ... ])\n"     + \
                 "   ( 5 -> F/[(3 -> <4>) \n" + \
                 "             (5 -> <6>) \n" + \
                 "              ... \n"       + \
                 "              ... ])"

        self.assertEqual(sn, sn_ref)

        # Format with newline and no cutoff

        sns = f"{a:n*}"

        sns_ref = "F/[( 2 -> F/[(2 -> <3>) \n" + \
                  "             (4 -> <5>) \n" + \
                  "             (6 -> <7>) \n" + \
                  "             (8 -> <9>) ])\n" + \
                  "   ( 5 -> F/[(3 -> <4>) \n" + \
                  "             (5 -> <6>) \n" + \
                  "             (7 -> <8>) ])"

        self.assertEqual(sns, sns_ref)

        # Format coord and payload and with newline and no cutoff

        snscp = f"{a:(02,03)n*}"

        snscp_ref = "F/[( 02 -> F/[(02 -> <003>) \n" + \
                    "              (04 -> <005>) \n" + \
                    "              (06 -> <007>) \n" + \
                    "              (08 -> <009>) ])\n" + \
                    "   ( 05 -> F/[(03 -> <004>) \n" + \
                    "              (05 -> <006>) \n" + \
                    "              (07 -> <008>) ])"

        self.assertEqual(snscp, snscp_ref)

        sr = f"{a!r}"

        sr_ref = "Fiber([2, 5], [Fiber([2, 4, 6, 8], [3, 5, 7, 9]), Fiber([3, 5, 7], [4, 6, 8])])"
        self.assertEqual(sr, sr_ref)


    def test_print_3D_flattened(self):
        """Test str format 3D flattened"""

        c0 = [2, 4, 6, 8]
        p0 = [3, 5, 7, 9]
        f0 = Fiber(c0, p0)

        c1 = [3, 5, 7]
        p1 = [4, 6, 8]
        f1 = Fiber(c1, p1)

        c = [(0, 2), (1, 5)]

        a = Fiber(c, [f0, f1])

        ss = f"{a:n*}"

        ss_ref = "F/[( (0, 2) -> F/[(2 -> <3>) \n" + \
                 "                  (4 -> <5>) \n" + \
                 "                  (6 -> <7>) \n" + \
                 "                  (8 -> <9>) ])\n" + \
                 "   ( (1, 5) -> F/[(3 -> <4>) \n" + \
                 "                  (5 -> <6>) \n" + \
                 "                  (7 -> <8>) ])"

        self.assertEqual(ss, ss_ref)

        sr = f"{a!r}"

        sr_ref = "Fiber([(0, 2), (1, 5)], [Fiber([2, 4, 6, 8], [3, 5, 7, 9]), Fiber([3, 5, 7], [4, 6, 8])])"
        self.assertEqual(sr, sr_ref)

    def test_print_lazy(self):
        """Test str format only works in eager mode"""

        c = [2, 4, 6, 8]
        p = [3, 5, 7, 9]

        a = Fiber(c, p)
        a._setIsLazy(True)

        self.assertEqual(str(a), "(Fiber, fromIterator)")

        with self.assertRaises(AssertionError):
            repr(a)

    def test_fiber2dict_eager_only(self):
        """Test that fiber2dict requires eager mode"""
        c = [2, 4, 6, 8]
        p = [3, 5, 7, 9]

        a = Fiber(c, p)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.fiber2dict()


if __name__ == '__main__':
    unittest.main()

