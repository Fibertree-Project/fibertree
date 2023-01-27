import os
import unittest

from fibertree import *

class TestFiber(unittest.TestCase):

    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

        # Make sure we have a tmp directory to write to
        if not os.path.exists("tmp"):
            os.makedirs("tmp")


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

    def test_default_eager(self):
        """Create a 1d fiber, built eagerly"""

        a = Fiber([2, 4, 6], [3, 5, 7])
        self.assertFalse(a.isLazy())

    def test_comparison_eq_ne(self):

        a = Fiber([2, 4, 6], [3, 5, 7])
        b = Fiber([2, 4, 6], [3, 5, 7])
        c = Fiber([2, 5, 6], [3, 5, 7])
        d = Fiber([2, 4, 6], [3, 6, 7])

        self.assertTrue(a == b)
        self.assertTrue(a != c)
        self.assertTrue(a != d)


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

    def test_fromCoordPayloadList(self):

        cp = [(2, 3), (4, 5), (6, 7)]

        (coords, payloads) = zip(*cp)

        a_ref = Fiber(coords=coords, payloads=payloads)

        a1 = Fiber.fromCoordPayloadList(*cp)
        self.assertEqual(a1, a_ref)
        self.assertEqual(a1.getDefault(), 0)

        # Removed functionality to set fiber default

#       a2 = Fiber.fromCoordPayloadList(*cp, default=1)
#       self.assertEqual(a2, a_ref)
#       self.assertEqual(a2.getDefault(), 1)

#       a3 = Fiber.fromCoordPayloadList(default=2, *cp)
#       self.assertEqual(a3, a_ref)
#       self.assertEqual(a3.getDefault(), 2)


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

        f = Fiber.fromUncompressed([0, 0, 0, 0, 0])

        self.assertEqual(f, f_ref)

    def test_fromUncompressed_2D_empty(self):
        """Create empty tensor from uncompressed 2-D"""

        f_ref = Fiber([], [])

        f = Fiber.fromUncompressed([[0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])

        self.assertEqual(f, f_ref)

    def test_fromRandom_2D(self):
        """Create a random 2D tensor"""

        shape = [10, 10]

        fiber_ref = Fiber.fromUncompressed([[0, 10, 10, 1, 0, 9, 8, 0, 0, 3],
                                            [9, 1, 0, 10, 1, 0, 10, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 3, 0, 3, 5, 0, 5, 7, 0, 0],
                                            [6, 0, 0, 0, 0, 0, 6, 0, 0, 0],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 2, 8, 2, 3, 7, 0, 0, 10],
                                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                            [0, 0, 0, 4, 0, 2, 9, 4, 0, 5],
                                            [6, 3, 0, 8, 0, 10, 0, 9, 4, 0]])

        fiber = Fiber.fromRandom(shape, [0.5, 0.5], 10, seed=3)

        self.assertEqual(fiber, fiber_ref)


    def test_getCoords(self):
        """Extract coordinates"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)

        c = a.getCoords()

        self.assertEqual(c, c_ref)


    def test_getCoords_eager_only(self):
        """Cannot get coordinates for a lazy fiber"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.getCoords()


    def test_getPayloads(self):
        """Extract payloads"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)

        p = a.getPayloads()

        self.assertEqual(p, p_ref)

    def test_getPayloads_eager_only(self):
        """Extract payloads"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.getPayloads()

    def test_isempty_1D(self):
        """Test for empty fiber"""

        a = Fiber([], [])
        self.assertTrue(a.isEmpty())

        b = Fiber([0, 1], [0, 0])
        self.assertTrue(b.isEmpty())

        c = Fiber([0, 1], [0, 1])
        self.assertFalse(c.isEmpty())


    def test_isempty_eager_only(self):
        """Test for empty fiber only in eager mode"""

        a = Fiber([], [])
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.isEmpty()


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

    def test_nonempty_eager_only(self):
        """Get non-empty elements only in eager mode"""

        a = Fiber([], [])
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.nonEmpty()

    def test_setDefault(self):
        """Test setting defaults - unimplemented"""

        pass

    def test_setOwner(self):
        """Test setting owner - unimplemented"""

        pass

    def test_rank_attrs(self):
        """Test getting and setting rank attributes"""
        # Set directly
        f = Fiber()
        attrs = RankAttrs("Unknown").setDefault(Payload(0))
        self.assertEqual(f.getRankAttrs(), attrs)

        attrs0 = RankAttrs("K", shape=20)
        attrs0.setFormat("U").setDefault(3).setId("M")
        f.setRankAttrs(attrs0)

        attrs1 = RankAttrs("K", shape=20)
        attrs1.setFormat("U").setDefault(3).setId("M")
        self.assertEqual(f.getRankAttrs(), attrs1)

        # Set via the constructor
        f = Fiber(rank_attrs=attrs)
        self.assertEqual(f.getRankAttrs(), attrs)

        # Set via the rank
        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        t = Tensor.fromFiber(rank_ids=["K"], fiber=a)

        attrs = RankAttrs("K").setShape(10).setDefault(0)
        self.assertEqual(a.getRankAttrs(), attrs)

    def test_active_range(self):
        """Test getting and setting the active range"""
        # Test default
        attrs = RankAttrs("K", shape=20)
        f = Fiber(rank_attrs=attrs)
        self.assertEqual(f.getActive(), (0, 20))

        # Set range explicitly
        f.setActive((3, 7))
        self.assertEqual(f.getActive(), (3, 7))

        # Reset active range
        f.setActive(None)
        self.assertEqual(f.getActive(), (0, 20))

        # Set via the constructor
        f = Fiber(active_range=(3, 7))
        self.assertEqual(f.getActive(), (3, 7))


    def test_minCoord(self):
        """Find minimum coordinate"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        c_min = min(c_ref)

        a = Fiber(c_ref, p_ref)

        self.assertEqual(a.minCoord(), c_min)

    def test_minCoord_eager_only(self):
        """Find minimum coordinate only works in eager mode"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.minCoord()

    def test_maxCoord(self):
        """Find minimum coordinate"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        c_max = max(c_ref)

        a = Fiber(c_ref, p_ref)

        self.assertEqual(a.maxCoord(), c_max)


    def test_maxCoord_eager_only(self):
        """Find maximum coordinate only works in eager mode"""

        c_ref = [2, 4, 6]
        p_ref = [3, 5, 7]

        a = Fiber(c_ref, p_ref)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.minCoord()

    def test_minmaxCoord_empty(self):

        f = Fiber([], [])

        self.assertIsNone(f.minCoord())
        self.assertIsNone(f.maxCoord())


    def test_values_2D(self):
        """Count values in a 2-D fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(a.countValues(), 6)


    def test_count_values_eager_only(self):
        """Count values in a 2-D fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.countValues()


    def test_values_with_zero(self):
        """Count values in a 1-D fiber with an explict zero"""

        a = Fiber([1, 8, 9], [2, 0, 10])

        self.assertEqual(a.countValues(), 2)


    def test_iterOccupancy(self):
        """Test iteration over non-default elements of a fiber"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterOccupancy()):
            self.assertEqual(c, c0[i])
            self.assertEqual(p, p0[i])

    def test_iterOccupancy_start_pos_eager_only(self):
        """Test iterOccupancy start_pos only works with eager fibers"""
        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            for _ in a.iterOccupancy(start_pos=5):
                pass

    def test_iterOccupancy_start_pos(self):
        """Test iteration over non-default elements of a fiber with the
        start_pos parameter"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterOccupancy(start_pos=1)):
            self.assertEqual(c, c0[i + 1])
            self.assertEqual(p, p0[i + 1])

        self.assertEqual(a.getSavedPosStats(), (2, 1))
        self.assertEqual(a.getSavedPos(), 2)


    def test_iterOccupancy_uses(self):
        """Test that iterOccupancy emits the correct trace of uses"""
        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 6, 7, 10]
        a_k = Fiber(c0, p0)
        a_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_iterOccupancy_uses")
        Metrics.trace("K")
        for _ in a_k.iterOccupancy():
            pass
        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,1,0\n",
            "1,4,1\n",
            "2,5,2\n",
            "3,8,3\n",
            "4,9,4\n"
        ]

        with open("tmp/test_iterOccupancy_uses-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_iterShape(self):
        """Test iteration over a fiber's shape"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]

        c0_ans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_ans = [0, 2, 0, 0, 0, 0, 0, 0, 0, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterShape()):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

    def test_iterShape_eager_only(self):
        """Test iterShape only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterShape())


    def test_iterShapeRef(self):
        """Test iteration over a fiber's shape with allocation"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]

        c0_ans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_ans = [0, 2, 0, 0, 0, 0, 0, 0, 0, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterShapeRef()):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_ans)
            self.assertEqual(a.payloads, p0_ans)

    def test_iterShapeRef_eager_only(self):
        """Test iterShapeRef only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterShapeRef())

    def test_iterActive(self):
        """Test iteration over non-default elements of a fiber within the active range"""

        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 7, 10]
        a = Fiber(c0, p0, active_range=(2, 9))

        c1 = [4, 8]
        p1 = [3, 7]

        for i, (c, p) in enumerate(a.iterActive()):
            self.assertEqual(c, c1[i])
            self.assertEqual(p, p1[i])

    def test_iterActive_start_pos_eager_only(self):
        """Test iterActive start_pos only works with eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 7, 10]

        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            for _ in a.iterActive(start_pos=5):
                pass

    def test_iterActive_start_pos(self):
        """Test iteration over non-default elements of a fiber with the
        start_pos parameter"""

        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 6, 7, 10]

        a = Fiber(c0, p0, active_range=(2, 9))

        c0_ans = [5, 8]
        p0_ans = [6, 7]

        for i, (c, p) in enumerate(a.iterActive(start_pos=2)):
            self.assertEqual(c, c0_ans[i])
            self.assertEqual(p, p0_ans[i])

        self.assertEqual(a.getSavedPosStats(), (2, 1))
        self.assertEqual(a.getSavedPos(), 3)


    def test_iterActive_uses(self):
        """Test that iterActive emits the correct trace of uses"""
        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 6, 7, 10]
        a_k = Fiber(c0, p0, active_range=(2, 9))
        a_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_iterActive_uses")
        Metrics.trace("K")
        for _ in a_k.iterActive():
            pass
        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,4,1\n",
            "1,5,2\n",
            "2,8,3\n"
        ]

        with open("tmp/test_iterActive_uses-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)


    def test_iterActiveShape(self):
        """Test iteration over the coordinates within the active shape"""

        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]

        c0_ans = [2, 3, 4, 5, 6, 7, 8]
        p0_ans = [0, 0, 3, 0, 0, 0, 0]

        a = Fiber(c0, p0, active_range=(2, 9))

        for i, (c, p) in enumerate(a.iterActiveShape()):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

    def test_iterActiveShape_eager_only(self):
        """Test iterActiveShape only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterActiveShape())


    def test_iterActiveShapeRef(self):
        """Test iteration over the coordinates within the fiber's active range,
        creating a fiber if one does not exist"""

        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]

        c0_ans = [2, 3, 4, 5, 6, 7, 8]
        p0_ans = [0, 0, 3, 0, 0, 0, 0]

        a = Fiber(c0, p0, active_range=(2, 9))

        for i, (c, p) in enumerate(a.iterActiveShapeRef()):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        c0_final = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_final = [2, 0, 0, 3, 0, 0, 0, 0, 10]

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_final)
            self.assertEqual(a.payloads, p0_final)

    def test_iterActiveShapeRef_eager_only(self):
        """Test iterActiveShapeRef only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterActiveShapeRef())


    def test_iterRange(self):
        """Test iteration over non-default elements of a fiber within the given range"""

        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 7, 10]
        a = Fiber(c0, p0)

        c1 = [4, 8]
        p1 = [3, 7]

        for i, (c, p) in enumerate(a.iterRange(2, 9)):
            self.assertEqual(c, c1[i])
            self.assertEqual(p, p1[i])

    def test_iterRange_start_pos_eager_only(self):
        """Test iterRange start_pos only works with eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 7, 10]

        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            for _ in a.iterRange(2, 9, start_pos=5):
                pass

    def test_iterRange_start_pos(self):
        """Test iteration over non-default elements of a fiber with the
        start_pos parameter"""

        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 6, 7, 10]

        a = Fiber(c0, p0)

        c0_ans = [5, 8]
        p0_ans = [6, 7]

        for i, (c, p) in enumerate(a.iterRange(2, 9, start_pos=2)):
            self.assertEqual(c, c0_ans[i])
            self.assertEqual(p, p0_ans[i])

        self.assertEqual(a.getSavedPosStats(), (2, 1))
        self.assertEqual(a.getSavedPos(), 3)

    def test_iterRange_flattened(self):
        """iterRange flattened coordinates"""

        coords = [(0, 2), (0, 4), (0, 6), (0, 8), (0, 9),
                  (1, 2), (1, 5), (1, 6), (1, 7),
                  (2, 0)]

        payloads = [3, 5, 7, 9, 10, 13, 16, 17, 18, 21]

        a = Fiber(coords, payloads)

        startc = [(0, 4), (0, 3), (0, 5), (1, 3) , (0, 9), (1, 5)]
        end_coord = [(0, 6),(0, 6), (0, 9), (1, 5), (1, 3), (1, 8)]

        ans = [Fiber(coords[1:2], payloads[1:2]),
               Fiber(coords[1:2], payloads[1:2]),
               Fiber(coords[2:4], payloads[2:4]),
               Fiber([], []),
               Fiber(coords[4:6], payloads[4:6]),
               Fiber(coords[6:9], payloads[6:9]),
        ]

        for i in range(len(startc)):
            class test_iterator:
                def __iter__(self):
                    return a.iterRange(startc[i], end_coord[i])

            c = Fiber.fromIterator(test_iterator, active_range=(startc[i], end_coord[i]))
            self.assertEqual(c, ans[i])

    def test_iterRange_uses(self):
        """Test that iterRange emits the correct trace of uses"""
        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 6, 7, 10]
        a_k = Fiber(c0, p0)
        a_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_iterRange_uses")
        Metrics.trace("K")
        for _ in a_k.iterRange(2, 9):
            pass
        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,4,1\n",
            "1,5,2\n",
            "2,8,3\n"
        ]

        with open("tmp/test_iterRange_uses-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)


    def test_iterRangeShape(self):
        """Test iteration over the coordinates within the given range"""

        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]

        c0_ans = [2, 4, 6, 8]
        p0_ans = [0, 3, 0, 0]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterRangeShape(2, 9, 2)):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

    def test_iterRangeShape_eager_only(self):
        """Test iterRangeShape only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterRangeShape(2, 9))


    def test_iterRangeShapeRef(self):
        """Test iteration over the coordinates within the given range,
        creating a fiber if one does not exist"""

        c0 = [1, 4, 5, 8, 9]
        p0 = [2, 3, 7, 0, 10]

        c0_ans = [2, 4, 6, 8]
        p0_ans = [0, 3, 0, 0]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a.iterRangeShapeRef(2, 9, 2)):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        c0_final = [1, 2, 4, 5, 6, 8, 9]
        p0_final = [2, 0, 3, 7, 0, 0, 10]

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_final)
            self.assertEqual(a.payloads, p0_final)

    def test_iterRangeShapeRef_eager_only(self):
        """Test iterRangeShapeRef only works in eager mode"""

        c0 = [1, 8, 9]
        p0 = [2, 0, 10]
        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            next(a.iterRangeShapeRef(2, 9))

    def test_coiterShape(self):
        """Test coiterShape"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)

        c0_ans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_ans = [(0, 3), (2, 0), (0, 4), (0, 0), (3, 0), (0, 7), (0, 0), (0, 0), (0, 0), (10, 1)]

        for i, (c, p) in enumerate(Fiber.coiterShape([a, b])):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

            self.assertEqual(b.coords, c1)
            self.assertEqual(b.payloads, p1)

    def test_coiterShape_eager_only(self):
        """Test coiterShape only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterShape([a, b])

    def test_coiterShapeRef(self):
        """Test coiterShapeRef"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5]
        p1 = [3, 4, 7]
        b = Fiber(c1, p1)

        c0_ans = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_ans = [(0, 3), (2, 0), (0, 4), (0, 0), (3, 0), (0, 7), (0, 0), (0, 0), (0, 0), (10, 0)]

        for i, (c, p) in enumerate(Fiber.coiterShapeRef([a, b])):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        c0_after = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_after = [0, 2, 0, 0, 3, 0, 0, 0, 0, 10]

        c1_after = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p1_after = [3, 0, 4, 0, 0, 7, 0, 0, 0, 0]
        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_after)
            self.assertEqual(a.payloads, p0_after)

            self.assertEqual(b.coords, c1_after)
            self.assertEqual(b.payloads, p1_after)

    def test_coiterShapeRef_eager_only(self):
        """Test coiterShapeRef only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterShapeRef([a, b])

    def test_coiterActiveShape(self):
        """Test coiterActiveShape"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)
        a.setActive((2, 9))

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)

        c0_ans = [2, 3, 4, 5, 6, 7, 8]
        p0_ans = [(0, 4), (0, 0), (3, 0), (0, 7), (0, 0), (0, 0), (0, 0)]

        for i, (c, p) in enumerate(Fiber.coiterActiveShape([a, b])):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

            self.assertEqual(b.coords, c1)
            self.assertEqual(b.payloads, p1)

    def test_coiterActiveShape_eager_only(self):
        """Test coiterActiveShape only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterActiveShape([a, b])

    def test_coiterActiveShapeRef(self):
        """Test coiterActiveShapeRef"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)
        a.setActive((2, 9))

        c1 = [0, 2, 5]
        p1 = [3, 4, 7]
        b = Fiber(c1, p1)

        c0_ans = [2, 3, 4, 5, 6, 7, 8]
        p0_ans = [(0, 4), (0, 0), (3, 0), (0, 7), (0, 0), (0, 0), (0, 0)]

        for i, (c, p) in enumerate(Fiber.coiterActiveShapeRef([a, b])):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        c0_after = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p0_after = [2, 0, 0, 3, 0, 0, 0, 0, 10]

        c1_after = [0, 2, 3, 4, 5, 6, 7, 8]
        p1_after = [3, 4, 0, 0, 7, 0, 0, 0]
        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_after)
            self.assertEqual(a.payloads, p0_after)

            self.assertEqual(b.coords, c1_after)
            self.assertEqual(b.payloads, p1_after)

    def test_coiterActiveShapeRef_eager_only(self):
        """Test coiterActiveShapeRef only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)
        a.setActive((2, 9))

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterActiveShapeRef([a, b])

    def test_coiterRangeShape(self):
        """Test coiterRangeShape"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)
        a.getRankAttrs().setId("K")

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)

        c0_ans = [2, 4, 6, 8]
        p0_ans = [(0, 4), (3, 0), (0, 0), (0, 0)]

        ans = Fiber.coiterRangeShape([a, b], 2, 9, 2)
        for i, (c, p) in enumerate(ans):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0)
            self.assertEqual(a.payloads, p0)

            self.assertEqual(b.coords, c1)
            self.assertEqual(b.payloads, p1)

            self.assertEqual(ans.getActive(), (2, 9))
            self.assertEqual(ans.getRankAttrs().getId(), "K")

    def test_coiterRangeShape_eager_only(self):
        """Test coiterRangeShape only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterRangeShape([a, b], 2, 9)

    def test_coiterRangeShapeRef(self):
        """Test coiterRangeShapeRef"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)
        a.getRankAttrs().setId("K")

        c1 = [0, 2, 5]
        p1 = [3, 4, 7]
        b = Fiber(c1, p1)

        c0_ans = [2, 4, 6, 8]
        p0_ans = [(0, 4), (3, 0), (0, 0), (0, 0)]

        ans = Fiber.coiterRangeShapeRef([a, b], 2, 9, 2)
        for i, (c, p) in enumerate(ans):
            with self.subTest(test=f"Element {i}"):
                self.assertEqual(c, c0_ans[i])
                self.assertEqual(p, p0_ans[i])
                self.assertIsInstance(p, Payload)

        c0_after = [1, 2, 4, 6, 8, 9]
        p0_after = [2, 0, 3, 0, 0, 10]

        c1_after = [0, 2, 4, 5, 6, 8]
        p1_after = [3, 4, 0, 7, 0, 0]
        with self.subTest(test="Test fiber internals"):
            self.assertEqual(a.coords, c0_after)
            self.assertEqual(a.payloads, p0_after)

            self.assertEqual(b.coords, c1_after)
            self.assertEqual(b.payloads, p1_after)

            self.assertEqual(ans.getActive(), (2, 9))
            self.assertEqual(ans.getRankAttrs().getId(), "K")

    def test_coiterRangeShapeRef_eager_only(self):
        """Test coiterRangeShapeRef only works on eager fibers"""
        c0 = [1, 4, 8, 9]
        p0 = [2, 3, 0, 10]
        a = Fiber(c0, p0)

        c1 = [0, 2, 5, 9]
        p1 = [3, 4, 7, 1]
        b = Fiber(c1, p1)
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            Fiber.coiterRangeShapeRef([a, b], 2, 9)

    def test_iter_no_fmt(self):
        """Test iteration over a fiber (default: iterOccupancy)"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a):
            self.assertEqual(c, c0[i])
            self.assertEqual(p, p0[i])

    def test_iter_no_fmt(self):
        """Test iteration over a fiber (default: iterOccupancy)"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)

        for i, (c, p) in enumerate(a):
            self.assertEqual(c, c0[i])
            self.assertEqual(p, p0[i])


    def test_iter_compressed(self):
        """Test iteration over a fiber (default: iterOccupancy)"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        t = Tensor.fromFiber(rank_ids=["K"], fiber=a)
        t.setFormat("K", "C")

        for i, (c, p) in enumerate(a):
            self.assertEqual(c, c0[i])
            self.assertEqual(p, p0[i])


    def test_iter_uncompressed(self):
        """Test iteration over a fiber (default: iterOccupancy)"""

        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        t = Tensor.fromFiber(rank_ids=["K"], fiber=a)
        t.setFormat("K", "U")

        c1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        p1 = [0, 2, 0, 0, 0, 0, 0, 0, 7, 10]

        for i, (c, p) in enumerate(a):
            self.assertEqual(c, c1[i])
            self.assertEqual(p, p1[i])


    def test_reversed_eager_only(self):
        """ Iterate reversed through eager fibers only """
        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a[::-1]


    def test_clear(self):
        """ Test clear """
        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        a.clear()

        b = Fiber()

        self.assertEqual(a, b)

    def test_clear_lazy(self):
        """ Test clear does not care if lazy"""
        c0 = [1, 8, 9]
        p0 = [2, 7, 10]

        a = Fiber(c0, p0)
        a._setIsLazy(True)
        a.clear()

        b = Fiber()

        self.assertEqual(a, b)
        self.assertFalse(a.isLazy())

    def test_getitem_index_error(self):
        Z_MN = Tensor(rank_ids=["M", "N"])
        z_m = Z_MN.getRoot()

        with self.assertRaises(IndexError) as cm:
            z_m[0]

        self.assertEqual(str(cm.exception), "The index (0) is out of range")

    def test_getitem_type_error(self):
        Z_MN = Tensor(rank_ids=["M", "N"])
        z_m = Z_MN.getRoot()

        with self.assertRaises(TypeError) as cm:
            z_m["foo"]

        self.assertEqual(str(cm.exception), "Invalid key type.")

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

    def test_getitem_eager_only(self):
        """Get item - eager mode only"""

        c_ref = [2, 4, 6, 8]
        p_ref = [3, 5, 7, 9]

        a = Fiber(c_ref, p_ref)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a[0]


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


    def test_getitem_nD(self):
        """Get item - multi-dimensional"""

        c00 = [1, 2, 3]
        p00 = [2, 3, 4]
        f00 = Fiber(c00, p00)

        c01 = [4, 6, 8]
        p01 = [5, 7, 9]
        f01 = Fiber(c01, p01)

        c02 = [5, 7]
        p02 = [6, 8]
        f02 = Fiber(c02, p02)

        c0 = [4, 5, 8]
        p0 = [f00, f01, f02]
        f = Fiber(c0, p0)

        f_1_1 = f[1, 1]
        f_1_1_ref = CoordPayload(c0[1], f01[1])

        self.assertEqual(f_1_1, f_1_1_ref)

        f_02_1 = f[0:2, 1]
        f_02_1_ref = Fiber(c0[0:2], [f00[1], f01[1]])

        self.assertEqual(f_02_1, f_02_1_ref)

        f_12_1 = f[1:2, 1]
        f_12_1_ref = Fiber(c0[1:2], [f01[1]])

        self.assertEqual(f_12_1, f_12_1_ref)

        f_02_01 = f[0:2, 0:1]
        f_02_01_ref = Fiber(c0[0:2], [f00[0:1], f01[0:1]])

        self.assertEqual(f_02_01, f_02_01_ref)

        f_13_02 = f[1:3, 0:2]
        f_13_02_ref = Fiber(c0[1:3], [f01[0:2], f02[0:2]])

        self.assertEqual(f_13_02, f_13_02_ref)

        f_13_12 = f[1:3, 1:2]
        f_13_12_ref = Fiber(c0[1:3], [f01[1:2], f02[1:2]])

        self.assertEqual(f_13_12, f_13_12_ref)

    def test_setitem_scalar(self):
        """test_setitem_scalar"""

        f = Fiber([0,1,3], [1,0,4])

        newf = Fiber([], [])

        newcoords = [ None, 0, 1, 2, 3, 4 ]
        newpayloads = [ 6, (4, 8), newf, None]

        ans_c = [0, 1, 3]
        ans_p = [6, (4, 8), newf, newf]

        for i in range(len(f)):
            for j, p in enumerate(newpayloads):
                f[i] = p
                a = f[i]
                self.assertEqual(a.coord, ans_c[i])
                self.assertEqual(a.payload, ans_p[j])

    def test_setitem_eager_only(self):
        """test_setitem - eager mode only"""

        f = Fiber([0,2,3], [1,0,4])
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f[0] = (1, 3)


    def test_setitem_coordpayload(self):
        """test_setitem_coordpayload"""

        f = Fiber([0,1,3], [1,0,4])

        newf = Fiber([], [])

        newcoords = [ None, 0, 1, 2, 3, 4 ]
        newpayloads = [ 6, (4, 8), newf, None]

        #
        # Dimensions position, newcoords-index, newpayload-index
        #
        ans_cvv = [[[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None]],
                   [[1, 1, 1, 1],
                    [None, None, None, None],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [None, None, None, None],
                    [None, None, None, None]],
                   [[3, 3, 3, 3],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4]]]

        ans_pvv = [[[6, (4, 8), newf, newf],
                    [6, (4, 8), newf, newf],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None]],
                   [[6, (4, 8), newf, newf],
                    [None, None, None, None],
                    [6, (4, 8), newf, newf],
                    [6, (4, 8), newf, newf ],
                    [None, None, None, None],
                    [None, None, None, None]],
                   [[6, (4, 8), newf, newf],
                    [None, None, None, None],
                    [None, None, None, None],
                    [None, None, None, None],
                    [6, (4, 8), newf, newf],
                    [6, (4, 8), newf, newf]]]

        for i in range(len(f)):

            for j, c in enumerate(newcoords):
                for k, p in enumerate(newpayloads):
                    a = f[i]
                    if ans_cvv[i][j][k] is not None:
                        f[i] = CoordPayload(c, p)
                        b = f[i]
                        self.assertEqual(b.coord, ans_cvv[i][j][k])
                        self.assertEqual(b.payload, ans_pvv[i][j][k])
                    else:
                        with self.assertRaises(CoordinateError):
                            f[i] = CoordPayload(c, p)


    def test_len(self):
        """Find length of a fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(len(a), 2)


    def test_len_lazy(self):
        """Find length of a fiber only when we are in eager mode"""
        class test_len_lazy_iterator:
            def __iter__(self):
                for i in range(5):
                    yield i * 2, i + 1

        a = Fiber.fromIterator(test_len_lazy_iterator)
        self.assertEqual(len(a), 5)


    def test_getPayload(self):
        """Access payloads"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 6, 3]
        answer_allocate = [0, 5, 7, 0]
        answer_noallocate = [None, 5, 7, None]
        answer_default = [-1, 5, 7, -1]

        for i in range(len(test)):
            self.assertEqual(a.getPayload(test[i]),
                             answer_allocate[i])
            self.assertEqual(a.getPayload(test[i], allocate=True),
                             answer_allocate[i])
            self.assertEqual(a.getPayload(test[i], allocate=False),
                             answer_noallocate[i])
            self.assertEqual(a.getPayload(test[i], allocate=False, default=-1),
                             answer_default[i])

    def test_getPayload_start_pos(self):
        """Get payload with a shortcut"""
        coords = [2, 4, 6]
        payloads = [3, 5, 7]
        a = Fiber(coords, payloads)

        # Basic test
        self.assertEqual(a.getPayload(2, start_pos=0), 3)
        self.assertEqual(a.getSavedPosStats(), (1, 0))
        self.assertEqual(a.getSavedPos(), 0)

        # Missing coordinate
        self.assertEqual(a.getPayload(5, start_pos=0), 0)
        self.assertEqual(a.getSavedPosStats(), (1, 2))
        self.assertEqual(a.getSavedPos(), 1)

        # Start at start_pos > 0
        self.assertEqual(a.getPayload(6, start_pos=1), 7)
        self.assertEqual(a.getSavedPosStats(), (1, 1))
        self.assertEqual(a.getSavedPos(), 2)

        # Empty fiber
        b = Fiber()
        self.assertEqual(b.getPayload(3, start_pos=0), 0)
        self.assertEqual(b.getSavedPosStats(), (1, 0))
        self.assertEqual(b.getSavedPos(), 0)


    def test_getPayload_start_pos_only_one_coord(self):
        """Ensure that getPayload only works if one coordinate is passed"""
        a = Fiber(default=Fiber)

        with self.assertRaises(AssertionError):
            a.getPayload(5, 7, start_pos=0)

    def test_getPayload_start_pos_less_than_coord(self):
        """Esure that getPayload must start before or at the coordinate"""
        coords = [2, 4, 6]
        payloads = [3, 5, 7]
        a = Fiber(coords, payloads)

        # If start_pos=0, this does not apply
        self.assertEqual(a.getPayload(0, start_pos=0), 0)

        with self.assertRaises(AssertionError):
            a.getPayload(3, start_pos=1)


    def test_getPayload_eager_only(self):
        """Can only access coordinates by payload in eager mode"""
        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.getPayload(3)


    def test_getPayloadRef_update(self):
        """Update payload references"""

        #
        # Test that each payload or allocated payload is an unique object
        # but updates do not get reflected back to the original fiber
        #
        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 5, 6, 3]
        update = [10, 11, 12, 13, 14]
        answer = [0, 5, 0, 7, 0]

        for i in range(len(test)):
            x = a.getPayload(test[i])
            x <<= update[i]
            self.assertEqual(a.getPayload(test[i]), answer[i])

    def test_getPayloadRef_eager_only(self):
        """Can only access coordinates by payload in eager mode"""
        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.getPayloadRef(3)

    def test_getPayload_2(self):
        """Access payloads - multilevel"""

        a = Fiber.fromUncompressed([[1, 2, 0, 4, 5, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 3, 4, 0, 0]])

        # Simple test
        self.assertEqual(a.getPayload(2, 2), 3)

        # Multiple tests
        test = [(0, 0), (2, 2), (0, 3), (2, 1)]
        answer_allocate = [1, 3, 4, 0]
        answer_noallocate = [1, 3, 4, None]

        for i in range(len(test)):
            p = a.getPayload(*test[i])
            self.assertEqual(p, answer_allocate[i])
            p = a.getPayload(*test[i], allocate=True)
            self.assertEqual(p, answer_allocate[i])
            p = a.getPayload(*test[i], allocate=False)
            self.assertEqual(p, answer_noallocate[i])

    def test_getPayload_2_update(self):
        """Update payloads - multilevel"""

        a = Fiber.fromUncompressed([[1, 2, 0, 4, 5, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 3, 4, 0, 0]])


        test = [(0,), (1,), (2, 0), (2, 2), (1, 1)]
        update = [ Fiber([3], [20]), Fiber([4], [21]), 22, 23, 24 ]
        check = [(0,3), (1, 3), (2,0), (2,2), (1,1)]
        answer = [20, 0, 0, 23, 0]

        for i in range(len(test)):
            with self.subTest(test=i):
                p = a.getPayload(*test[i], allocate=True)
                p <<= update[i]
                q = a.getPayload(*check[i])
                self.assertEqual(q, answer[i])



    def test_getPayload_3(self):
        """Access payloads - complex"""

        a = Fiber.fromUncompressed([[1, 2, 0, 4, 5, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 3, 4, 0, 0]])

        a_1 = Fiber([], [])
        a_2 = Fiber([2, 3],[3, 4])

        # Simple test
        self.assertEqual(a.getPayload(2), a_2)

        # Multiple tests
        test = [(2,), (1,), (1, 2)]
        answer_allocate = [a_2, a_1, 0 ]
        answer_noallocate = [a_2, None, None ]
        answer_default = [a_2, -1, -1]

        for i in range(len(test)):
            p = a.getPayload(*test[i])
            self.assertEqual(p, answer_allocate[i])
            p = a.getPayload(*test[i], allocate=True)
            self.assertEqual(p, answer_allocate[i])
            p = a.getPayload(*test[i], allocate=False)
            self.assertEqual(p, answer_noallocate[i])
            p = a.getPayload(*test[i], allocate=False, default=-1)
            self.assertEqual(p, answer_default[i])


    def test_getPayload_shortcut(self):
        """getPayload with shortcut"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 6, 3, 6]
        start_pos = [0, 0, 1, 0, Payload(2)]

        answer_saved_pos = [0, 1, 2, 0, 2]
        answer_saved_stats = [(1, 0),
                              (2, 1),
                              (3, 2),
                              (4, 3),
                              (5, 3)]


        for i in range(len(test)):
            p = a.getPayload(test[i], start_pos=start_pos[i])
            saved_pos = a.getSavedPos()
            saved_pos_stats = a.getSavedPosStats(clear=False)
            self.assertEqual(saved_pos, answer_saved_pos[i])
            self.assertEqual(saved_pos_stats, answer_saved_stats[i])


    def test_getPayloadRef(self):
        """Get payload references"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 6, 3]
        answer = [0, 5, 7, 0]

        for i in range(len(test)):
            self.assertEqual(a.getPayloadRef(test[i]), answer[i])

    def test_getPayloadRef_update(self):
        """Update payload references"""

        #
        # Test that each payload or allocated payload is an unique object
        #
        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        test = [0, 4, 5, 6, 3]
        update = [10, 11, 12, 13, 14]
        answer = [10, 11, 12, 13, 14]

        for i in range(len(test)):
            x = a.getPayloadRef(test[i])
            x <<= update[i]
            self.assertEqual(a.getPayload(test[i]), answer[i])


    def test_getPayloadRef2(self):
        """Get payload references 2-D"""

        t = Tensor(rank_ids=["m", "n"])
        a = t.getRoot()

        test = [(0,), (2,), (1, 3), (2, 1)]
        answer = [Fiber([], []), Fiber([], []), 0, 0]

        for i in range(len(test)):
            p = a.getPayloadRef(*test[i])
            self.assertEqual(p, answer[i])

        with self.assertRaises(AssertionError):
            a.getPayloadRef(3, 2, 4)

    def test_getPayloadRef2_update(self):
        """Update payload references 2-D"""

        t = Tensor(rank_ids=["m", "n"])
        a = t.getRoot()

        test = [(0,), (2,), (1, 3), (2, 1)]
        update = [ Fiber([3], [20]), Fiber([4], [21]), 22, 23 ]
        check = [(0,3), (2,4), (1,3), (2,1)]
        answer = [20, 21, 22, 23]

        for i in range(len(test)):
            with self.subTest(test=i):
                p = a.getPayloadRef(*test[i])
                p <<= update[i]
                q = a.getPayload(*check[i])
                self.assertEqual(q, answer[i])


    def test_getPayloadRef_shortcut(self):
        """getPayloadRef with shortcut"""

        # TBD: Fill in this test...

        pass

    def test_ilshift(self):
        """<<= infix operator"""

        coords = [2, 4, 6, 8, 9, 12, 15, 16, 17, 20 ]
        payloads = [3, 5, 7, 9, 10, 13, 16, 17, 18, 21]

        a = Fiber(coords, payloads)
        b = Fiber()

        b <<= a

        self.assertEqual(a, b)

    def test_ilshift_multiple_ranks(self):
        """ <<= infix operator with multiple ranks"""
        a = Fiber.fromUncompressed([[0, 1, 0, 3], [0, 0, 0, 0], [8, 0, 0, 0]])
        b = Fiber()

        b <<= a

        self.assertEqual(a, b)

    def test_ilshift_eager_only(self):
        """<<= infix operator eager mode only"""

        coords = [2, 4, 6, 8, 9, 12, 15, 16, 17, 20 ]
        payloads = [3, 5, 7, 9, 10, 13, 16, 17, 18, 21]

        a = Fiber(coords, payloads)
        b = Fiber()
        b._setIsLazy(True)

        with self.assertRaises(AssertionError):
            b <<= a

    def test_append(self):
        """Append element at end of fiber"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)

        aa_coords = [2, 4, 6, 7]
        aa_payloads = [3, 5, 7, 10]

        aa_ref = Fiber(aa_coords, aa_payloads)

        retval = a.append(7, 10)

        self.assertIsNone(retval)
        self.assertEqual(a, aa_ref)

    def test_append_eager_only(self):
        """Append element at end of fiber"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads)
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.append(7, 10)

    def test_append_empty(self):
        """Append to empty fiber"""

        a = Fiber([], [])
        a_ref = Fiber( [4], [8])

        retval = a.append(4, 8)

        self.assertIsNone(retval)
        self.assertEqual(a, a_ref)


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

        retval = a.extend(b)

        self.assertIsNone(retval)
        self.assertEqual(a, ae_ref)

    def test_extend_eager_only(self):
        """Extend fiber only in eager mode"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]

        a = Fiber(a_coords, a_payloads)
        a._setIsLazy(True)

        b_coords = [7, 10, 12]
        b_payloads = [4, 6, 8]

        b = Fiber(b_coords, b_payloads)

        with self.assertRaises(AssertionError):
            a.extend(b)


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

    def test_concat_eager_only(self):
        """Concat should force both fibers to be eager"""
        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]
        a = Fiber(a_coords, a_payloads)

        b_coords = [6, 10, 12]
        b_payloads = [4, 6, 8]
        b = Fiber(b_coords, b_payloads)

        a._setIsLazy(True)
        with self.assertRaises(AssertionError):
            a.concat(b)

        a._setIsLazy(False)
        b._setIsLazy(True)
        with self.assertRaises(AssertionError):
            a.concat(b)

#    def test_insert(self):
#        """Insert payload at coordinates 0, 3, 7"""
#
#        coords = [2, 4, 6]
#        payloads = [3, 5, 7]
#
#        a = Fiber(coords, payloads)
#
#        insert_at = [0, 3, 7]
#
#        ans = {}
#        ans[0] = Fiber([0, 2, 4, 6], [1, 3, 5, 7])
#        ans[3] = Fiber([0, 2, 3, 4, 6], [1, 3, 10, 5, 7])
#        ans[7] = Fiber([0, 2, 3, 4, 6, 7], [1, 3, 10, 5, 7, 50])
#
#        for i in insert_at:
#            p = i*i+1
#            retval = a.insert(i, p)
#
#            self.assertIsNone(retval)
#            self.assertEqual(a, ans[i])


    def test_shape(self):
        """Test determining shape of a fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        s = a.getShape()

        self.assertEqual(s, [5, 8])

    def test_shape_empty(self):
        """Test determining shape of an empty fiber"""

        a = Fiber([], [])

        s = a.getShape()

        self.assertEqual(s, [0])

    def test_shape_authoritative(self):
        """Test the authoritative parameter for getShape"""
        a = Fiber([1, 2, 3], [3, 4, 5])
        s = a.getShape(all_ranks=False, authoritative=True)
        self.assertIsNone(s)

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml", shape=5)
        s = a.getShape(all_ranks=False, authoritative=True)
        self.assertEqual(s, 5)

    def test_estimateShape_eager_only(self):
        """Test determining shape of a fiber, eager mode only"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.estimateShape()

    def test_rankids(self):
        """Test finding rankids of a fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        r = a.getRankIds()

        self.assertEqual(r, ["X.1", "X.0"])

    def test_getDepth_eager_only(self):
        """Test getDepth for eager only"""
        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.getDepth()

    def test_uncompress(self):
        """Test uncompress"""

        uncompressed_ref = [[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 2, 0, 0, 5, 0, 0, 8],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 3, 0, 5, 0, 7, 0]]


        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        uncompressed = a.uncompress()

        self.assertEqual(uncompressed, uncompressed_ref)

    def test_uncompress_eager_only(self):
        """Test uncompress, eager mode only"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        a._setIsLazy(True)

        with self.assertRaises(AssertionError):
            a.uncompress()

    def test_uncompress_default(self):
        """Test uncompress with non-zero default"""

        uncompressed_ref = [[-1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, -1, -1, -1, -1, -1],
                            [-1,  2, -1, -1,  5, -1, -1,  8],
                            [-1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1,  3, -1,  5, -1,  7, -1]]


        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")
        # Dirty way of setting non-zero default...
        for c, p in a:
            p._setDefault(-1)

        uncompressed = a.uncompress()

        self.assertEqual(uncompressed, uncompressed_ref)

    def test_project(self):
        """Test projections"""

        c = [0, 1, 10, 20]
        p = [1, 2, 11, 21]
        a = Fiber(c, p)
        a.setActive((0, 25))

        cp = [1, 2, 11, 21]
        ap_ref = Fiber(cp, p)

        ap = a.project(lambda c: c + 1)

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), (1, 26))

    def test_project_interval(self):
        """Test project when given an interval"""
        c = [0, 1, 10, 20]
        p = [1, 2, 11, 21]
        a = Fiber(c, p)
        a.setActive((0, 25))

        cp = [1, 2, 11]
        ap_ref = Fiber(cp, p[:3])

        ap = a.project(lambda c: c + 1, interval=(1, 20))

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), (1, 20))

    def test_project_reverse(self):
        """Test project when the transfer function reverses the coordinates"""
        c = [0, 1, 10, 20]
        p = [1, 2, 11, 21]
        a = Fiber(c, p)
        a.setActive((0, 25))

        cp = [20, 40, 58, 60]
        ap_ref = Fiber(cp, list(reversed(p)))

        ap = a.project(lambda c: 60 - 2 * c)

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), (12, 61))

    def test_project_coord_ex(self):
        """Test whether project works on tuple coordinates with an explicit
        coordinate example"""
        c = [(0, 1), (0, 3), (1, 4)]
        p = [1, 2, 3]

        a = Fiber(c, p, active_range=((0, 0), (2, 5)))

        cp = [1, 3, 5]
        ap_ref = Fiber(cp, p)

        ap = a.project(lambda c: (c[0] + c[1]), coord_ex=(0, 0))

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), (0, 6))

    def test_project_coord_ex_default(self):
        """Test whether project works on tuple coordinates with a default
        coordinate example"""
        c = [(0, 1), (0, 3), (1, 4)]
        p = [1, 2, 3]

        a = Fiber(c, p, active_range=((0, 0), (2, 5)))

        cp = [1, 3, 5]
        ap_ref = Fiber(cp, p)

        ap = a.project(lambda c: (c[0] + c[1]))

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), (0, 6))

    def test_project_tup_out(self):
        """Test whether project works on tuple coordinates in the output"""
        c = [1, 4, 5]
        p = [1, 2, 3]

        a = Fiber(c, p, shape=6)

        cp = [(1, 2), (4, 5), (5, 6)]
        ap_ref = Fiber(cp, p)

        ap = a.project(lambda c: (c, c + 1))

        self.assertEqual(ap, ap_ref)
        self.assertEqual(ap.getActive(), ((0, 1), (6, 7)))

    def test_project_reverse_eager_only(self):
        """Test projections, eager only if reversed"""

        class test_project_reverse_eager_only_iterator:
            def __init__(self):
                self.c = [0, 1, 10, 20]
                self.p = [1, 2, 11, 21]

            def __iter__(self):
                for c, p in zip(self.c, self.p):
                    yield c, p

        a = Fiber.fromIterator(test_project_reverse_eager_only_iterator)

        with self.assertRaises(AssertionError):
            ap = a.project(lambda c: 50 - c)

    def test_project_start_pos_eager_only(self):
        """Test projections, eager only if start_pos"""

        class test_project_start_pos_eager_only_iterator:
            def __init__(self):
                self.c = [0, 1, 10, 20]
                self.p = [1, 2, 11, 21]

            def __iter__(self):
                for c, p in zip(self.c, self.p):
                    yield c, p

        a = Fiber.fromIterator(test_project_start_pos_eager_only_iterator)

        with self.assertRaises(AssertionError):
            ap = a.project(lambda c: 10 + c, start_pos=2)

    def test_project_use_stats(self):
        """Test that use stats are tracked correctly after project"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_project_use_stats")
        Metrics.trace("I")
        for _ in f_k.project(trans_fn=lambda k: k + 1, rank_id="I"):
            pass
        Metrics.endCollect()

        corr = [
            "I_pos,I,fiber_pos\n",
            "0,3,0\n",
            "1,5,1\n",
            "2,7,2\n",
            "3,9,3\n"
        ]

        with open("tmp/test_project_use_stats-I-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_project_collecting_requires_rank_id(self):
        """Test that project requires non-None rank_id"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        Metrics.beginCollect()
        with self.assertRaises(AssertionError):
            f_i = f_k.project(lambda k: k + 1)
            next(f_i.__iter__())
        Metrics.endCollect()

    def test_project_rank_id_correct_no_collect(self):
        """Test that the rank_id is correctly set to Unknown if no collection
        occurs"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        f_i = f_k.project(lambda k: k + 1)
        self.assertEqual(f_i.getRankAttrs().getId(), "Unknown")

    def test_project_trace_dst_in_loop_order(self):
        """Test that project works correctly if the destination is in the loop
        order"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_project_trace_dst_in_loop_order")
        Metrics.trace("K", "project_0")
        for m, p in f_k.project(trans_fn=lambda k: k + 3, rank_id="M", start_pos=1):
            pass
        Metrics.endCollect()

        corr = [
            "M_pos,M,fiber_pos\n",
            "0,4,1\n",
            "1,6,2\n",
            "2,8,3\n"
        ]

        with open("tmp/test_project_trace_dst_in_loop_order-K-project_0.csv") as f:
            self.assertEqual(f.readlines(), corr)

    def test_project_trace_src_in_loop_order(self):
        """Test that project works correctly if the source is in the loop
        order"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_project_trace_src_in_loop_order")
        Metrics.trace("K", "project_0")

        for m, p in \
                f_k.project(
                    trans_fn=lambda k: k + 3, rank_id="M", interval=(6,100),
                    tick=True
                ).iterOccupancy(tick=False):
            pass
        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,4,1\n",
            "1,6,2\n",
            "2,8,3\n"
        ]

        with open("tmp/test_project_trace_src_in_loop_order-K-project_0.csv") as f:
            self.assertEqual(f.readlines(), corr)

    def test_prune(self):
        """Test pruning a fiber"""

        f = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f.setActive((0, 10))

        fl2_ref = Fiber([2, 4], [4, 8])
        fu2_ref = Fiber([6, 8], [12, 16])

        #
        # Prune out lower half
        #
        f0 = f.prune(lambda n, c, p: n < 2)
        self.assertEqual(f0, fl2_ref)

        # Check that active_range does not change
        self.assertEqual(f0.getActive(), (0, 10))

        f1 = f.prune(lambda n, c, p: c < 5)
        self.assertEqual(f1, fl2_ref)

        f2 = f.prune(lambda n, c, p: p < 10)
        self.assertEqual(f2, fl2_ref)

        #
        # Prune out upper half
        #
        f3 = f.prune(lambda n, c, p: n >= 2)
        self.assertEqual(f3, fu2_ref)

        f4 = f.prune(lambda n, c, p: c > 5)
        self.assertEqual(f4, fu2_ref)

        f5 = f.prune(lambda n, c, p: p > 10)
        self.assertEqual(f5, fu2_ref)

        #
        # Prune out lower half and stop
        #
        f6 = f.prune(lambda n, c, p: True if p < 10 else None)
        self.assertEqual(f6, fl2_ref)

    def test_prune_use_stats(self):
        """Test that use stats are tracked correctly after prune"""
        f_k = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_prune_use_stats")
        Metrics.trace("K")
        for _ in f_k.prune(lambda i, c, p: i % 2 == 0):
            pass
        Metrics.endCollect()

        corr = [
            "K_pos,K,fiber_pos\n",
            "0,2,0\n",
            "1,6,1\n"
        ]

        with open("tmp/test_prune_use_stats-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

    def test_getPosition_eager_only(self):
        """getPosition only works in eager mode"""
        f = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f.getPosition(4)

    def test_getPosition(self):
        """Basic getPosition test"""
        f = Fiber([2, 4, 6, 8], [4, 8, 12, 16])

        self.assertEqual(f.getPosition(4), 1)
        self.assertEqual(f.getPosition(5), None)

    def test_getPositionRef_eager_only(self):
        """getPositionRef only works in eager mode"""
        f = Fiber([2, 4, 6, 8], [4, 8, 12, 16])
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f.getPositionRef(4)

    def test_getPositionRef(self):
        """Basic getPositionRef test"""
        f = Fiber([2, 4, 6, 8], [4, 8, 12, 16])

        self.assertEqual(f.getPositionRef(4), 1)
        self.assertEqual(f.getPositionRef(5), 2)

        corr = Fiber([2, 4, 5, 6, 8], [4, 8, 0, 12, 16])
        self.assertEqual(f, corr)

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

    def test_unzip_eager_only(self):
        """Test unzipping only works for eager Fibers"""
        c = [0, 1, 10, 20]
        p_ab = [(0, 1), (1, 2), (10, 11), (20, 21)]

        ab = Fiber(c, p_ab)
        ab._setIsLazy(True)

        with self.assertRaises(AssertionError):
            ab.unzip()

    def test_updateCoords(self):
        """Update coords"""

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
        flat_split = split.flattenRanks()
        flat_split.updateCoords(lambda i, c, p: c[1])

        self.assertEqual(f, flat_split)


    def test_updateCoords_eager_only(self):
        """Update coords only for eager mode"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f.updateCoords(lambda i, c, p: c + 1)


    def test_updateCoords_reversed(self):
        """Update coords - where coordinates need to be reversed"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)

        f_ans = Fiber([ 100-c for c in reversed(c)], list(reversed(p)))

        f.updateCoords(lambda i, c, p: 100-c)

        self.assertEqual(f, f_ans)


    def test_updatePayloads_eager_only(self):
        """Update payloads only for eager mode"""

        #
        # Create the fiber to be split
        #
        c = [0, 1, 9, 10, 12, 31, 41]
        p = [ 0, 10, 20, 100, 120, 310, 410 ]

        f = Fiber(c,p)
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f.updatePayloads(lambda p: p + 1)


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

    def test_iadd_eager_only(self):
        """iadd fibers, eager mode only"""

        a_coords = [2, 4, 6]
        a_payloads = [3, 5, 7]
        a = Fiber(a_coords, a_payloads)
        a._setIsLazy(True)

        b_coords = [7, 10, 12]
        b_payloads = [4, 6, 8]
        b = Fiber(b_coords, b_payloads)

        with self.assertRaises(AssertionError):
            a += b


    def test_and(self):
        """Intersection test"""

        a = Fiber([1, 5, 8, 9], [2, 6, 9, 10])
        b = Fiber([0, 5, 9], [2, 7, 11])
        ab = a & b

        ff = Fiber([5, 9], [(6, 7), (10, 11)])

        for test, corr in zip(ab, ff):
            self.assertEqual(test, corr)

    def test_and_empty(self):
        """Intersection test - with explict zeros"""

        a = Fiber([1, 5, 8, 9], [0, 6, 0, 10])
        b = Fiber([1, 5, 8, 9], [2, 0, 0, 11])

        ab = a & b

        ff = Fiber([9], [(10, 11)])

        for test, corr in zip(ab, ff):
            self.assertEqual(test, corr)


    def test_or(self):
        """Union test"""

        a = Fiber([1, 5, 8, 9], [2, 6, 9, 10])
        a.getRankAttrs().setId("K").setShape(20)
        b = Fiber([0, 5, 9], [2, 7, 11])
        b.getRankAttrs().setId("K").setShape(20)

        ab_ref = Fiber([0, 1, 5, 8, 9],
                       [("B", 0, 2),
                        ("A", 2, 0),
                        ("AB", 6, 7),
                        ("A", 9, 0),
                        ("AB", 10, 11)])

        ab = a | b

        self.assertEqual(ab, ab_ref)

        # Check the fiber attributes
        self.assertEqual(ab.getRankAttrs().getId(), "K")
        self.assertEqual(ab.getActive(), (0, 20))

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

    def test_or_2d(self):
        """Union test 2d"""

        a1 = [[1, 2, 3, 0],
              [1, 0, 3, 4],
              [0, 2, 3, 4],
              [1, 2, 0, 4]]

        a2 = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        a3 = [[2, 3, 4, 5],
              [0, 0, 0, 0],
              [1, 0, 3, 4],
              [1, 2, 0, 4]]

        b1 = a2
        b2 = a1
        b3 = a3

        au = [a1, a2, a3]
        bu = [b1, b2, b3]

        a = Fiber.fromUncompressed(au)
        b = Fiber.fromUncompressed(bu)

        x = a|b

        ab_ref = ["A", "B", "AB"]
        a1_fiber = Fiber.fromUncompressed(a1)
        a2_fiber = Fiber([],[])
        a3_fiber = Fiber.fromUncompressed(a3)

        ab_a_ref = [a1_fiber, a2_fiber, a3_fiber]
        ab_b_ref = [a2_fiber, a1_fiber, a3_fiber]

        for n, (c, (ab, ab_a, ab_b)) in enumerate(x):
            self.assertEqual(ab, ab_ref[n])
            self.assertEqual(ab_a, ab_a_ref[n])
            self.assertEqual(ab_b, ab_b_ref[n])


    def test_xor(self):
        """Xor test"""

        a = Fiber([1, 5, 8, 9], [2, 6, 9, 10])
        a.getRankAttrs().setId("K").setShape(10)
        b = Fiber([0, 5, 9], [2, 7, 11])
        b.getRankAttrs().setId("K").setShape(10)

        ab_ref = Fiber([0, 1, 8],
                       [("B", 0, 2),
                        ("A", 2, 0),
                        ("A", 9, 0)])

        ab = a ^ b

        self.assertEqual(ab, ab_ref)
        self.assertEqual(ab.getRankAttrs().getId(), "K")
        self.assertEqual(ab.getActive(), (0, 10))

    def test_xor_2d(self):
        """Union test 2d"""

        a1 = [[1, 2, 3, 0],
              [1, 0, 3, 4],
              [0, 2, 3, 4],
              [1, 2, 0, 4]]

        a2 = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        a3 = [[2, 3, 4, 5],
              [0, 0, 0, 0],
              [1, 0, 3, 4],
              [1, 2, 0, 4]]

        b1 = a2
        b2 = a1
        b3 = a3

        au = [a1, a2, a3]
        bu = [b1, b2, b3]
        abu_ref = [a1, b2, []]

        a = Fiber.fromUncompressed(au)
        b = Fiber.fromUncompressed(bu)

        x = a ^ b

        ab_ref = ["A", "B"]
        a1_fiber = Fiber.fromUncompressed(a1)
        a2_fiber = Fiber([],[])

        ab_a_ref = [a1_fiber, a2_fiber]
        ab_b_ref = [a2_fiber, a1_fiber]

        for n, (c, (ab, ab_a, ab_b)) in enumerate(x):
            self.assertEqual(ab, ab_ref[n])
            self.assertEqual(ab_a, ab_a_ref[n])
            self.assertEqual(ab_b, ab_b_ref[n])



    def test_xor_empty(self):
        """Uniontest - with explict zeros"""

        a = Fiber([1, 5, 8, 9], [0, 6, 0, 10])
        b = Fiber([1, 5, 8, 9], [2, 0, 0, 11])

        ab_ref = Fiber([1, 5],
                       [("B", 0, 2),
                        ("A", 6, 0)])

        ab = a ^ b

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
        ab = a << b
        ff = Fiber([1, 5, 8, 9], [(0, 2), (10, 6), (0, 9), (0, 10)])

        for test, corr in zip(ab, ff):
            self.assertEqual(test, corr)

    def test_flatten(self):
        """Test flattening/unflattening 1 level"""

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

        ff = f.flattenRanks()

        ff_ref = Fiber([(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)],
                       [Fiber([0, 1, 2], [1, 2, 3]),
                        Fiber([0, 2, 3], [1, 3, 4]),
                        Fiber([1, 2, 3], [2, 3, 4]),
                        Fiber([0, 1, 3], [1, 2, 4]),
                        Fiber([0, 1, 2], [1, 2, 3]),
                        Fiber([0, 2, 3], [1, 3, 4]),
                        Fiber([0, 1, 3], [1, 2, 4])])

        self.assertEqual(ff, ff_ref)
        self.assertEqual(ff.getShape(), [(3, 4), 4])
        self.assertEqual(ff.getActive(), ((0, 0), (3, 4)))

        fu = ff.unflattenRanks()

        self.assertEqual(fu, f)

    def test_flatten_default_correct(self):
        """Test flattening/unflattening 1 level"""

        u_t = [[1, 2, 3, 0],
               [1, 0, 3, 4],
               [0, 2, 3, 4],
               [1, 2, 0, 4]]

        f = Fiber.fromUncompressed(u_t)

        for _, f2 in f:
            f2._setDefault(10)

        ff = f.flattenRanks()
        ff_ref = Fiber([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (1, 3),
                        (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 3)],
                        [1, 2, 3, 1, 3, 4, 2, 3, 4, 1, 2, 4])

        self.assertEqual(ff, ff_ref)
        self.assertEqual(ff.getDefault(), 10)

        fu = ff.unflattenRanks()

        self.assertEqual(fu, f)
        self.assertEqual(fu.getDefault(), Fiber)

        for _, f2 in fu:
            self.assertEqual(f2.getDefault(), 10)

    def test_flatten_deepcopies(self):
        """Test flattening/unflattening 1 level"""

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

        ff = f.flattenRanks()

        ff_ref = Fiber([(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)],
                       [Fiber([0, 1, 2], [1, 2, 3]),
                        Fiber([0, 2, 3], [1, 3, 4]),
                        Fiber([1, 2, 3], [2, 3, 4]),
                        Fiber([0, 1, 3], [1, 2, 4]),
                        Fiber([0, 1, 2], [1, 2, 3]),
                        Fiber([0, 2, 3], [1, 3, 4]),
                        Fiber([0, 1, 3], [1, 2, 4])])

        self.assertEqual(ff, ff_ref)

        changed = f.getPayload(0, 0)
        changed.append(3, 1)

        self.assertEqual(ff, ff_ref)

    def test_flatten_payload_error(self):
        """PayloadError is raised if the payloads of the given fiber are not fibers"""
        f = Fiber.fromUncompressed([1, 2, 0, 0, 4, 0])

        with self.assertRaises(PayloadError):
            f.flattenRanks()

    def test_flatten_unflatten_eager_only(self):
        """Test flattening/unflattening, eager only"""

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
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f.flattenRanks()

        ff = Fiber([(0, 0), (0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 3)],
                    [Fiber([0, 1, 2], [1, 2, 3]),
                     Fiber([0, 2, 3], [1, 3, 4]),
                     Fiber([1, 2, 3], [2, 3, 4]),
                     Fiber([0, 1, 3], [1, 2, 4]),
                     Fiber([0, 1, 2], [1, 2, 3]),
                     Fiber([0, 2, 3], [1, 3, 4]),
                     Fiber([0, 1, 3], [1, 2, 4])])
        ff._setIsLazy(True)

        with self.assertRaises(AssertionError):
            ff.unflattenRanks()

    def test_flatten_levels_2(self):
        """Test flattening/unflattening 2 levels"""

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

        ff = f.flattenRanks(levels=2)

        ref_coords = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0),
                      (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 2),
                      (0, 2, 3), (0, 3, 0), (0, 3, 1), (0, 3, 3),
                      (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0),
                      (2, 1, 2), (2, 1, 3), (2, 3, 0), (2, 3, 1),
                      (2, 3, 3)]

        ref_payloads = [1, 2, 3, 1, 3, 4, 2, 3, 4, 1, 2, 4, 1, 2,
                        3, 1, 3, 4, 1, 2, 4]

        ff_ref = Fiber(coords=ref_coords, payloads=ref_payloads)

        self.assertEqual(ff, ff_ref)

        #
        # Now unflatten back to the original
        #
        fu = ff.unflattenRanks(levels=2)

        self.assertEqual(fu, f)

        #
        # Now unflatten in two steps
        #
        fu1 = ff.unflattenRanks(levels=1)
        fu1.updatePayloads(lambda i, c, p: p.unflattenRanks(levels=1))

        self.assertEqual(fu1, f)

    def test_flatten_levels_3(self):
        """Test flattening/unflattening 3 levels"""

        # TBD
        pass

    def test_flatten_style_linear(self):
        """Test the linear flattening style"""
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

        ff = f.flattenRanks(levels=2, style="linear")

        ref_coords = [0, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 15,
                      32, 33, 34, 36, 38, 39, 44, 45, 47]

        ref_payloads = [1, 2, 3, 1, 3, 4, 2, 3, 4, 1, 2, 4, 1, 2,
                        3, 1, 3, 4, 1, 2, 4]

        ff_ref = Fiber(coords=ref_coords, payloads=ref_payloads)

        self.assertEqual(ff, ff_ref)

    def test_merge(self):
        """Test that mergeRanks merges together fibers"""
        f = Fiber([0, 1, 4, 5],
                  [Fiber([0, 1, 2], [1, 2, 3], shape=10),
                   Fiber([1, 3, 4], [4, 5, 6], shape=10),
                   Fiber([4, 7], [7, 8], shape=10),
                   Fiber([5, 7], [9, 10], shape=10)],
                  shape=10)
        mf = f.mergeRanks(style="absolute")

        corr = Fiber([0, 1, 2, 3, 4, 5, 7], [1, 6, 3, 5, 13, 9, 18])
        self.assertEqual(mf, corr)
        self.assertEqual(mf.getShape(), [10])
        self.assertEqual(mf.getActive(), (0, 10))

    def test_merge_max(self):
        """Test mergeRanks with a custom merge function"""
        f = Fiber([0, 1, 4, 5],
                  [Fiber([0, 1, 2], [1, 2, 3], shape=10),
                   Fiber([1, 3, 4], [4, 5, 6], shape=10),
                   Fiber([4, 7], [7, 8], shape=10),
                   Fiber([5, 7], [9, 10], shape=10)],
                  shape=10)
        mf = f.mergeRanks(style="absolute", merge_fn=lambda ps: max(ps))

        corr = Fiber([0, 1, 2, 3, 4, 5, 7], [1, 4, 3, 5, 7, 9, 10])
        self.assertEqual(mf, corr)
        self.assertEqual(mf.getShape(), [10])
        self.assertEqual(mf.getActive(), (0, 10))

    def test_merge_two_levels(self):
        """Test merging more than one level"""
        f = Fiber([0, 4],
                  [Fiber([0, 1],
                         [Fiber([0, 1, 2], [1, 2, 3], shape=10),
                          Fiber([1, 3, 4], [4, 5, 6], shape=10)],
                         shape=10),
                   Fiber([4, 5],
                         [Fiber([4, 7], [7, 8], shape=10),
                          Fiber([5, 7], [9, 10], shape=10)],
                         shape=10)],
                  shape=10)
        mf = f.mergeRanks(levels=2, style="absolute")

        corr = Fiber([0, 1, 2, 3, 4, 5, 7], [1, 6, 3, 5, 13, 9, 18])
        self.assertEqual(mf, corr)
        self.assertEqual(mf.getShape(), [10])
        self.assertEqual(mf.getActive(), (0, 10))

    def test_merge_depth_gt_zero(self):
        """Test that we can merge tensors with depth > 1"""
        f = Fiber([0, 4],
                  [Fiber([0, 1],
                         [Fiber([0, 1, 2], [1, 2, 3], shape=10),
                          Fiber([1, 3, 4], [4, 5, 6], shape=10)],
                         shape=10),
                   Fiber([4, 5],
                         [Fiber([4, 7], [7, 8], shape=10),
                          Fiber([5, 7], [9, 10], shape=10)],
                         shape=10)],
                  shape=5)
        mf = f.mergeRanks(depth=1, style="absolute")

        corr = Fiber([0, 4],
                     [Fiber([0, 1, 2, 3, 4], [1, 6, 3, 5, 6]),
                      Fiber([4, 5, 7], [7, 9, 18])])
        self.assertEqual(mf, corr)
        self.assertEqual(mf.getShape(), [5, 10])
        self.assertEqual(mf.getActive(), (0, 5))

    def test_merge_fibers(self):
        """Test that the merge works to correctly combine fibers"""
        f = Fiber([0, 4],
                  [Fiber([0, 1],
                         [Fiber([0, 1, 2], [1, 2, 3], shape=10),
                          Fiber([1, 3, 4], [4, 5, 6], shape=10)],
                         shape=10),
                   Fiber([1, 5],
                         [Fiber([4, 7], [7, 8], shape=10),
                          Fiber([5, 7], [9, 10], shape=10)],
                         shape=10)],
                  shape=7)
        mf = f.mergeRanks(style="absolute")

        corr = Fiber([0, 1, 5],
                     [Fiber([0, 1, 2], [1, 2, 3]),
                      Fiber([1, 3, 4, 7], [4, 5, 13, 8]),
                      Fiber([5, 7], [9, 10])])

        self.assertEqual(mf, corr)
        self.assertEqual(mf.getShape(), [10, 10])
        self.assertEqual(mf.getActive(), (0, 10))


if __name__ == '__main__':
    unittest.main()
