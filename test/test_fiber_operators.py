""" Tests of fiber operators """

import unittest

from fibertree import Fiber
from fibertree import Payload

class TestFiberOperators(unittest.TestCase):
    """ Tests of fiber operators """

    def test_add_int(self):
        """Test __add__ integers"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber.fromUncompressed([3, 4, 5, 2, 2, 8])

        with self.subTest("f_in + 2"):
            f_out = f_in + 2
            self.assertEqual(f_ref, f_out)

        with self.subTest("2 + f_in"):
            f_out = 2 + f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in += 2"):
            # f_in gets clobbered!
            f_in += 2
            self.assertEqual(f_ref, f_in)


    def test_add_payload(self):
        """Test __add__ payload"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber.fromUncompressed([3, 4, 5, 2, 2, 8])
        two = Payload(2)

        with self.subTest("f_in + 2"):
            f_out = f_in + two
            self.assertEqual(f_ref, f_out)

        with self.subTest("2 + f_in"):
            f_out = two + f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in += 2"):
            # f_in gets clobbered!
            f_in += two
            self.assertEqual(f_ref, f_in)


    def test_add_fiber(self):
        """Test __add__ fiber"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        g_in = Fiber([6, 8], [20, 22])
        fg_ref = Fiber([0, 1, 2, 5, 6, 8], [1, 2, 3, 6, 20, 22])

        with self.subTest("f+g"):
            fg_out = f_in + g_in
            self.assertEqual(fg_ref, fg_out)

        with self.subTest("f+=g"):
            # f_in gets clobbered!
            f_in += g_in
            self.assertEqual(fg_ref, f_in)


    def test_mul_int(self):
        """Test __mul__ integers"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber([0, 1, 2, 5], [2, 4, 6, 12])

        with self.subTest("f_in * 2"):
            f_out = f_in * 2
            self.assertEqual(f_ref, f_out)

        with self.subTest("2*f_in"):
            f_out = 2 * f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in *=2"):
            # f_in gets clobbered!
            f_in *= 2
            self.assertEqual(f_ref, f_in)

    def test_mul_payload(self):
        """Test __mul__ payload"""

        f_in = Fiber.fromUncompressed([1, 2, 3, 0, 0, 6])
        f_ref = Fiber([0, 1, 2, 5], [2, 4, 6, 12])
        two = Payload(2)

        with self.subTest("f_in * 2"):
            f_out = f_in * two
            self.assertEqual(f_ref, f_out)

        with self.subTest("2*f_in"):
            f_out = two * f_in
            self.assertEqual(f_ref, f_out)

        with self.subTest("f_in *=2"):
            # f_in gets clobbered!
            f_in *= two
            self.assertEqual(f_ref, f_in)


if __name__ == '__main__':
    unittest.main()
