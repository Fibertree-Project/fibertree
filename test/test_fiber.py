import unittest
from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.tensor_image import TensorImage

class TestFiber(unittest.TestCase):

    def test_new_1d(self):
        """Create a 1d fiber"""

        a = Fiber([2, 4, 6], [3, 5, 7])

    def test_new_2d(self):
        """Create a 1d fiber"""

        b0 = Fiber([1, 4, 7], [2, 5, 8])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        a0 = Fiber([2, 4], [b0, b1])

    def test_new_empty(self):
        """Create an empty fiber"""

        a = Fiber([], [])

    def test_comparison_eq(self):

        a = Fiber([2, 4, 6], [3, 5, 7])
        b = Fiber([2, 4, 6], [3, 5, 7])

        self.assertEqual(a, b)

    def test_comparison_eq_1D(self):

        a = Fiber([2, 4, 6], [3, 5, 7])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        b2 = Fiber([2, 4, 6, 8], [3, 5, 7, 0])
        b3 = Fiber([2, 4, 6], [3, 6, 7])
        b4 = Fiber([2, 4, 8], [3, 5, 7])

        self.assertEqual(a, b1)
        self.assertEqual(a, b2)
        self.assertNotEqual(a, b3)
        self.assertNotEqual(a, b4)

        c = Fiber([], [])
        d1 = Fiber([0, 1], [0, 0])
        d2 = Fiber([0, 1], [0, 10])

        self.assertEqual(c, d1)
        self.assertNotEqual(c, d2)

    def test_comparison_eq_2D(self):

        a = Fiber([2, 4, 6], [3, 5, 7])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        b2 = Fiber([2, 4, 6, 8], [3, 5, 7, 0])
        b3 = Fiber([2, 4, 6], [3, 6, 7])
        b4 = Fiber([2, 4, 8], [3, 5, 7])

        x0 = Fiber([2, 4], [a, a])
        x1 = Fiber([2, 4], [a, b1])
        x2 = Fiber([2, 4], [a, b2])
        x3 = Fiber([2, 4], [a, b3])
        x4 = Fiber([2, 4], [a, b4])

        self.assertEqual(x0, x1)
        self.assertEqual(x0, x2)
        self.assertNotEqual(x0, x3)
        self.assertNotEqual(x0, x4)
        self.assertEqual(x1, x2)
        self.assertNotEqual(x1, x3)
        self.assertNotEqual(x1, x4)
        self.assertNotEqual(x2, x3)
        self.assertNotEqual(x2, x4)
        self.assertNotEqual(x3, x4)


    def test_fromYAMLfile_1D(self):
        """Read a YAMLfile 1-D"""

        a_ref = Fiber([2, 4, 6], [3, 5, 7])

        a = Fiber.fromYAMLfile("./data/test_fiber-1.yaml")

        self.assertEqual(a, a_ref)

    def test_fromYAMLfile_2D(self):
        """Read a YAMLfile 2-D"""

        b0 = Fiber([1, 4, 7], [2, 5, 8])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        a_ref = Fiber([2, 4], [b0, b1])

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(a, a_ref)

    def test_fromUncompressed_1D(self):
        """Create from uncompressed 1-D"""

        f_ref = Fiber([0, 1, 3, 4], [1, 2, 4, 5])

        f = Fiber.fromUncompressed([1, 2, 0, 4, 5, 0])

        self.assertEqual(f, f_ref)

    def test_fromUncompressed_2D(self):
        """Create from uncompressed 2-D"""

        a1 = Fiber([0, 1, 3, 4], [1, 2, 4, 5])
        a2 = Fiber([2, 3], [3, 4])

        f_ref = Fiber([0, 2], [a1, a2])

        f = Fiber.fromUncompressed([[1, 2, 0, 4, 5, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 3, 4, 0, 0]])

        self.assertEqual(f, f_ref)

    def test_fromUncompressed_3D(self):
        """Create from uncomrpessed 3-D"""

        f_ref = Fiber.fromYAMLfile("./data/test_fiber-3.yaml")

        u_t = [[[1, 2, 3, 0],
                [1, 0, 3, 4],
                [0, 2, 3, 4],
                [1, 2, 0, 4]],
               [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]],
               [[1, 2, 3, 0],
                [1, 0, 3, 4],
                [0, 0, 0, 0],
                [1, 2, 0, 4]]]

        f = Fiber.fromUncompressed(u_t)

        self.assertEqual(f, f_ref)

    def test_fromUncompressed_1D_empty(self):
        """Create empty tensor from uncompressed 1-D"""

        f_ref = Fiber([], [])

        f = Fiber.fromUncompressed([ 0, 0, 0, 0, 0 ])

        self.assertEqual(f, f_ref)

    def test_fromUncompressed_2D_empty(self):
        """Create empty tensor from uncompressed 2-D"""

        f_ref = Fiber([], [])

        f = Fiber.fromUncompressed([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])

        self.assertEqual(f, f_ref)


    def test_getCoords(self):
        """Extract coordinates"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)

        c = a.getCoords()

        self.assertEqual(c, c_ref)

    def test_getPayloads(self):
        """Extract payloads"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)

        p = a.getPayloads()

        self.assertEqual(p, p_ref)

    def test_isempty_1D(self):
        """Test for empty fiber"""

        a = Fiber([], [])
        self.assertTrue(a.isEmpty())

        b = Fiber([0, 1], [0, 0])
        self.assertTrue(b.isEmpty())

        c = Fiber([0, 1], [0, 1])
        self.assertFalse(c.isEmpty())

    def test_isempty_2D(self):
        """Test for empty fiber"""

        a1 = Fiber([], [])
        a2 = Fiber([0, 1], [0, 0])
        a3 = Fiber([0, 1], [0, 1])

        a = Fiber([2, 3], [a1, a1])
        self.assertTrue(a.isEmpty())

        b = Fiber([3, 4], [a2, a2])
        self.assertTrue(b.isEmpty())

        c = Fiber([3, 4], [a1, a2])
        self.assertTrue(c.isEmpty())

        d = Fiber([4, 5], [a1, a3])
        self.assertFalse(d.isEmpty())

    def test_nonempty_2D(self):
        """Test for empty fiber"""

        a1 = Fiber([], [])
        a2 = Fiber([0, 1], [0, 0])
        a3 = Fiber([0, 1], [0, 1])

        a = Fiber([1, 2, 3], [a1, a2, a3])

        ne = a.nonEmpty()

        ne3 = Fiber([1], [1])
        ne_ref = Fiber([3], [ne3])

        self.assertEqual(ne, ne_ref)

    def test_setDefault(self):
        """Test setting defaults - unimplemented"""

        pass

    def test_setOwner(self):
        """Test setting owner - unimplemented"""

        pass


    def test_minCoord(self):
        """Find minimum coordinate"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        c_min = min(c_ref)

        a = Fiber(c_ref, p_ref)

        self.assertEqual(a.minCoord(), c_min)


    def test_maxCoord(self):
        """Find minimum coordinate"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        c_max = max(c_ref)

        a = Fiber(c_ref, p_ref)

        self.assertEqual(a.maxCoord(), c_max)


    def test_values_2D(self):
        """Count values in a 2-D fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(a.countValues(), 6)


    def test_values_with_zero(self):
        """Count values in a 1-D fiber with an explict zero"""

        a = Fiber([1, 8, 9], [2, 0, 10])

        self.assertEqual(a.countValues(), 2)


    def test_getitem_simple(self):
        """Get item - simple"""

        c_ref = [2, 4, 6, 8]
        p_ref = [3, 5, 7, 9]

        a = Fiber(c_ref, p_ref)

        (coord0, payload0) = a[0]

        self.assertEqual(coord0, 2)
        self.assertEqual(payload0, 3)

        (coord1, payload1) = a[1]

        self.assertEqual(coord1, 4)
        self.assertEqual(payload1, 5)

        (coord2, payload2) = a[-2]
        self.assertEqual(coord2, 6)
        self.assertEqual(payload2, 7)

        (coord3, payload3) = a[-1]
        self.assertEqual(coord3, 8)
        self.assertEqual(payload3, 9)


    def test_getitem_slice(self):
        """Get item - slices"""

        c_ref = [2, 4, 6, 8]
        p_ref = [3, 5, 7, 9]

        a = Fiber(c_ref, p_ref)

        slice1 = a[0:2]

        slice1_coord_ref = a.coords[0:2]
        slice1_payload_ref = a.payloads[0:2]
        slice1_ref = Fiber(slice1_coord_ref, slice1_payload_ref)

        self.assertEqual(slice1, slice1_ref)


    def test_len(self):
        """Find lenght of a fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(len(a), 2)


    def test_getPayload(self):
        """Access payloads"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 6, 3]
        answer = [None, 5, 7, None]
        
        for i in range(len(test)):
            self.assertEqual(a.getPayload(test[i]), answer[i])

    def test_getPayload_2(self):
        """Access payloads - multilevel"""

        a = Fiber.fromUncompressed([[1, 2, 0, 4, 5, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 3, 4, 0, 0]])

        # Simple test
        self.assertEqual(a.getPayload(2, 2), 3)

        # Multiple tests
        test = [(0, 0), (2, 2), (1, 3), (2, 1)]
        answer = [1, 3, None, None]

        for i in range(len(test)):
            p = a.getPayload(*test[i])
            self.assertEqual(p, answer[i])


    def test_append(self):
        """Append element at end of fiber"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        aa_coords = [2, 4, 6, 7]
        aa_payloads = [3, 5, 7, 10]

        aa_ref = Fiber(aa_coords, aa_payloads)

        a.append(7, 10)

        self.assertEqual(a, aa_ref)

    def test_append_assert(self):
        """Append element at end of fiber - and assert"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        with self.assertRaises(AssertionError):
            a.append(3, 10)

    def test_extend(self):
        """Extend fiber"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]

        a = Fiber(a_coords, a_payloads)

        b_coords = [7, 10, 12]
        b_payloads = [4, 6, 8]

        b = Fiber(b_coords, b_payloads)

        ae_coords = [2, 4, 6, 7, 10, 12]
        ae_payloads = [3, 5, 7, 4, 6, 8]

        ae_ref = Fiber(ae_coords, ae_payloads)

        a.extend(b)

        self.assertEqual(a, ae_ref)


    def test_extend_assert(self):
        """Extend fiber - and assert"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]

        a = Fiber(a_coords, a_payloads)

        b_coords = [6, 10, 12]
        b_payloads = [4, 6, 8]

        b = Fiber(b_coords, b_payloads)

        with self.assertRaises(AssertionError):
            a.extend(b)

        with self.assertRaises(AssertionError):
            a.extend(1)


    def test_insert(self):
        """Insert payload at coordinates 0, 3, 7"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        insert_at = [0, 3, 7]

        ans = {}
        ans[0] = Fiber([0, 2, 4, 6], [1, 3, 5, 7])
        ans[3] = Fiber([0, 2, 3, 4, 6], [1, 3, 10, 5, 7])
        ans[7] = Fiber([0, 2, 3, 4, 6, 7], [1, 3, 10, 5, 7, 50])

        for i in insert_at:
            p = i*i+1
            a.insert(i, p)

            self.assertEqual(a, ans[i])


    def test_shape(self):

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        s = a.getShape()

        self.assertEqual(s, [5, 8])


    def test_uncompress(self):
        """Test recursive iteration"""

        uncompressed_ref = [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 5, 0, 0, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 5, 0, 7, 0]]


        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        uncompressed = a.uncompress()

        self.assertEqual(uncompressed, uncompressed_ref)


    def test_project(self):
        """Test projections"""

        c = [0, 1, 10, 20]
        p = [1, 2, 11, 21]
        a = Fiber(c, p)

        cp = [1, 2, 11, 21]
        ap_ref = Fiber(cp, p)

        ap = a.project(lambda c: c + 1)

        self.assertEqual(ap, ap_ref)


    def test_upzip(self):
        """Test unzipping a fiber"""

        c = [0, 1, 10, 20]
        p_a = [0, 1, 10, 20]
        p_b = [1, 2, 11, 21]

        p_ab = [(0, 1), (1, 2), (10, 11), (20, 21)]

        a_ref = Fiber(c, p_a)
        b_ref = Fiber(c, p_b)
        ab = Fiber(c, p_ab)

        (a, b) = ab.unzip()

        self.assertEqual(a, a_ref)
        self.assertEqual(b, b_ref)

    def test_add(self):
        """Add fibers"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]

        a = Fiber(a_coords, a_payloads)

        b_coords = [7, 10, 12]
        b_payloads = [4, 6, 8]

        b = Fiber(b_coords, b_payloads)

        ae_coords = [2, 4, 6, 7, 10, 12]
        ae_payloads = [3, 5, 7, 4, 6, 8]

        ae_ref = Fiber(ae_coords, ae_payloads)

        self.assertEqual(a+b, ae_ref)

    def test_iadd(self):
        """iadd fibers"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]

        a = Fiber(a_coords, a_payloads)

        b_coords = [7, 10, 12]
        b_payloads = [4, 6, 8]

        b = Fiber(b_coords, b_payloads)

        ae_coords = [2, 4, 6, 7, 10, 12]
        ae_payloads = [3, 5, 7, 4, 6, 8]

        ae_ref = Fiber(ae_coords, ae_payloads)

        a += b

        self.assertEqual(a, ae_ref)


    def test_and(self):
        """Intersection test"""

        a = Fiber([1, 5, 8, 9], [2, 6, 9, 10])
        b = Fiber([0, 5, 9], [2, 7, 11])

        ab_ref = Fiber([5, 9], [(6, 7), (10, 11)])

        ab = a & b

        self.assertEqual(ab, ab_ref)

    def test_and_empty(self):
        """Intersection test - with explict zeros"""

        a = Fiber([1, 5, 8, 9], [0, 6, 0, 10])
        b = Fiber([1, 5, 8, 9], [2, 0, 0, 11])

        ab_ref = Fiber([9], [(10, 11)])

        ab = a & b

        self.assertEqual(ab, ab_ref)


    def test_or(self):
        """Union test"""

        a = Fiber([1, 5, 8, 9], [2, 6, 9, 10])
        b = Fiber([0, 5, 9], [2, 7, 11])

        ab_ref = Fiber([0, 1, 5, 8, 9],
                       [("B", 0, 2),
                        ("A", 2, 0),
                        ("AB", 6, 7),
                        ("A", 9, 0),
                        ("AB", 10, 11)])

        ab = a | b

        self.assertEqual(ab, ab_ref)

    def test_or_empty(self):
        """Uniontest - with explict zeros"""

        a = Fiber([1, 5, 8, 9], [0, 6, 0, 10])
        b = Fiber([1, 5, 8, 9], [2, 0, 0, 11])

        ab_ref = Fiber([1, 5, 9],
                       [("B", 0, 2),
                        ("A", 6, 0),
                        ("AB", 10, 11)])

        ab = a | b

        self.assertEqual(ab, ab_ref)


    def test_diff(self):
        """Difference test"""

        a = Fiber([1, 5, 8, 9, 12, 14], [2, 6, 9, 10, 0, 0])
        b = Fiber([0, 5, 9, 12], [2, 7, 0, 5])

        # Notes:
        #     coordinate 9 stays since b@9 is zero
        #     coordinat 12 goes away even though explict zero at a@12
        #     coordinate 14 does not go away with explict zero at a@14


        ab_ref = Fiber([1, 8, 9, 14],
                       [2, 9, 10, 0])

        ab = a - b

        self.assertEqual(ab, ab_ref)


    def test_assignment(self):
        """Assignment test"""

        a = Fiber([0, 5, 9], [0, 10, 0])
        b = Fiber([1, 5, 8, 9, 14], [2, 6, 9, 10, 0])

        # Note:
        #    coordinate 9 stays although a@9 is zero
        #    coordinate 14 does not appear since b@14 is 0

        ab_ref = Fiber([1, 5, 8, 9],
                       [(0, 2),
                        (10, 6),
                        (0, 9),
                        (0, 10)])

        ab = a << b

        self.assertEqual(ab, ab_ref)



"""


    print("Union")

    a.print()
    b.print()

    ab = a | b
    ab.print()
    print("----\n\n")

    print("Assignment")

    z = Fiber()
    a = Fiber([2, 6, 8], [4, 8, 10])

    z.print("Z Fiber")
    a.print("A Fiber")

    za = z << a
    za.print("Z << A Fiber")
    print("----\n\n")
"""


if __name__ == '__main__':
    unittest.main()

