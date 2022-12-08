import unittest
from fibertree import CoordPayload
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor


class TestConstructor(unittest.TestCase):

    def setUp(self):

        self.input = {}

        self.input["c1"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.input["p1"] = [7, 1, 8, 3, 8, 4, 6, 3, 7, 5]

        self.input["c2"] = [ 0, 1, 2]
        self.input["p2"] = [ Fiber([2], [4]), Fiber([1], [4]), Fiber([2], [2])]


    def test_constructor_1D(self):
        """Test constructor 1D"""

        ans = Fiber([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [7, 1, 8, 3, 8, 4, 6, 3, 7, 5])

        attrs = []
        attrs.append([[0], 1, None, [10]])
        attrs.append([[-1], 1, None, [10]])
        attrs.append([[0], 1, None, [20]])
        attrs.append([[-2], 1, None, [20]])


        fs = []

        fs.append(Fiber(self.input["c1"], self.input["p1"]))
        fs.append(Fiber(self.input["c1"], self.input["p1"], default=-1))
        fs.append(Fiber(self.input["c1"], self.input["p1"], shape=20))
        fs.append(Fiber(self.input["c1"], self.input["p1"], shape=20, default=-2))

        for test, f in enumerate(fs):
            with self.subTest(test=test):
                if test != 2:
                    continue
                f_attr = self.attributes(f)

                for n, (c, p)  in enumerate(f):
                    self.assertEqual(c, self.input["c1"][n])
                    self.assertEqual(p, self.input["p1"][n])

                self.assertEqual(f, ans)
                self.assertEqual(f_attr, attrs[test])

    def test_constructor_copies_elements(self):
        """Test that the constructor copies the coordinate and payload lists"""
        c = [1, 3, 6]
        p = [2, 4, 9]
        a = Fiber(c, p)

        c.append(10)
        p.append(2)

        self.assertEqual(a.getCoords(), [1, 3, 6])
        self.assertEqual(a.getPayloads(), [2, 4, 9])


    def test_constructor_2D(self):
        """Test constructor 2D"""

        ans = Fiber([0, 1, 2],
                    [Fiber([2], [4]),
                     Fiber([1], [4]),
                     Fiber([2], [2])])

        #
        # Note: the default value does not get propaged down
        #       the fibertree.... so it doesn't make sense when
        #       fibers are inserted as the payloads.
        #
        attrs = []
        attrs.append([[Fiber, 0], 2, None, [3,3]])
        attrs.append([[Fiber, 0], 2, None, [3,3]])
        attrs.append([[Fiber, 0], 2, None, [8,3]])
        attrs.append([[Fiber, 0], 2, None, [8,3]])

        fs = []

        fs.append(Fiber(self.input["c2"], self.input["p2"]))
        fs.append(Fiber(self.input["c2"], self.input["p2"], default=-1))
        fs.append(Fiber(self.input["c2"], self.input["p2"], shape=8))
        fs.append(Fiber(self.input["c2"], self.input["p2"], shape=8, default=-2))


        for test, f in enumerate(fs):
            with self.subTest(test=test):
                f_attr = self.attributes(f)

                for n, (c, p)  in enumerate(f):
                    self.assertEqual(c, self.input["c2"][n])
                    self.assertEqual(p, self.input["p2"][n])

                self.assertEqual(f, ans)
                self.assertEqual(f_attr, attrs[test])


    def test_fromRandom_1D_dense(self):
        """Test random 1d sparse"""

        ans = Fiber([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [7, 1, 8, 3, 8, 4, 6, 3, 7, 5])

        attr = [[0], 1, None, [10]]

        f = Fiber.fromRandom([10], [1.0], 9, 10)
        f_attr = self.attributes(f)

        self.assertEqual(f, ans)
        self.assertEqual(f_attr, attr)


    def test_fromRandom_1D_sparse(self):
        """Test random 1d sparse"""

        ans = Fiber([3, 6, 8, 9, 12, 16, 19, 20, 28, 30, 32, 38, 40, 43, 46, 47, 48, 49],
                    [8, 9, 6, 3, 5, 4, 1, 4, 6, 4, 1, 6, 2, 6, 5, 9, 2, 5])

        attr = [[0], 1, None, [50]]

        f = Fiber.fromRandom([50], [0.3], 9, 10)
        f_attr = self.attributes(f)

        self.assertEqual(f, ans)
        self.assertEqual(f_attr, attr)


#        self.makeTest(f, f_attr)


    def test_fromRandom_2D_dense(self):
        """Test random 2d sparse"""

        ans = Fiber([0, 1, 2],
                    [Fiber([0, 1, 2], [1, 4, 2]),
                     Fiber([0, 1, 2], [1, 3, 2]),
                     Fiber([0, 1, 2], [3, 3, 3])])

        attr = [[Fiber, 0], 2, None, [3, 3]]

        f = Fiber.fromRandom([3, 3], [1.0, 1.0], 4, 10)
        f_attr = self.attributes(f)

        self.assertEqual(f, ans)
        self.assertEqual(f_attr, attr)


    def test_fromRandom_2D_sparse(self):
        """Test random 1d sparse"""

        ans = Fiber([0, 1, 2],
                    [Fiber([2], [4]),
                     Fiber([1], [4]),
                     Fiber([2], [2])])

        attr = [[Fiber, 0], 2, None, [3, 3]]

        f = Fiber.fromRandom([3, 3], [1.0, 0.3], 4, 10)
        f_attr = self.attributes(f)

        self.assertEqual(f, ans)
        self.assertEqual(f_attr, attr)

    def test_fromIterator(self):
        """Test fromIterator"""
        c = [2, 4, 6]
        p = [3, 5, 7]

        class iterator:
            coords = c
            payloads = p

            def __iter__(self):
                for coord, payload in zip(self.coords, self.payloads):
                    yield CoordPayload(coord, payload)

        f = Fiber.fromIterator(iterator)
        ff = Fiber(c, p)

        for p1, p2 in zip(f, ff):
            self.assertEqual(p1, p2)

    def test_fromLazy(self):
        """Test fromLazy"""
        c = [2, 4, 6]
        p = [3, 5, 7]

        class iterator:
            coords = c
            payloads = p

            def __iter__(self):
                for coord, payload in zip(self.coords, self.payloads):
                    yield CoordPayload(coord, payload)

        lazy = Fiber.fromIterator(iterator)
        lazy.getRankAttrs().setShape(7)
        eager = Fiber.fromLazy(lazy)

        self.assertEqual(eager.getCoords(), c)
        self.assertEqual(eager.getPayloads(), p)
        self.assertFalse(eager.isLazy())


    @staticmethod
    def makeTest(f, a):
        """Make a check for a test"""

#        self.makeTest(f, f_attr)

        print("")
        print(f"        ans = {f!r}")
        print(f"        attr = {a}")
        print("")


    @staticmethod
    def attributes(f):
        """Get all attributes of a fiber"""

        defaults = []

        ff = f
        while isinstance(ff, Fiber):
            defaults.append(ff.getDefault())
            ff = (ff.payloads or [Fiber([],[])])[0]


        attributes = [ defaults,
                       f.getDepth(),
                       f.getOwner(),
                       f.getShape() ]

        return attributes

if __name__ == '__main__':
    unittest.main()

