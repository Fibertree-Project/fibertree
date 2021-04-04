import unittest
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor


class TestUnionIntersect(unittest.TestCase):

    def setUp(self):

        self.input = {}

        self.input["a1_M"] = Tensor.fromUncompressed(["M"], [1, 0, 3, 0, 5, 0, 7])
        self.input["a1_m"] = self.input["a1_M"].getRoot()

        self.input["b1_M"] = Tensor.fromUncompressed(["M"], [2, 0, 4, 5])
        self.input["b1_m"] = self.input["b1_M"].getRoot()

        self.input['c1_M'] = Tensor.fromUncompressed(["M"], [1, 2, 3])
        self.input["c1_m"] = self.input["c1_M"].getRoot()

        self.input["a2_MK"] = Tensor.fromUncompressed(["M", "K"], [[1, 0, 3, 0, 5, 0, 7],
                                                                   [2, 2, 0, 3, 0, 0, 8],
                                                                   [0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0],
                                                                   [4, 0, 5, 0, 8, 0, 9]])

        self.input["a2_m"] = self.input["a2_MK"].getRoot()

        self.input["b2_MK"] = Tensor.fromUncompressed(["M", "K"], [[2, 0, 4, 5],
                                                                   [0, 0, 0, 0],
                                                                   [3, 4, 6, 0],
                                                                   [0, 0, 0, 0],
                                                                   [1, 2, 3, 4]])
        self.input["b2_m"] = self.input["b2_MK"].getRoot()


    def test_union_2x_1d(self):
        """Test union 2-way for 1d fibers"""

        ans = Fiber([0, 2, 3, 4, 6],
                    [('AB', Payload(1), Payload(2)),
                     ('AB', Payload(3), Payload(4)),
                     ('B', Payload(0), Payload(5)),
                     ('A', Payload(5), Payload(0)),
                     ('A', Payload(7), Payload(0))])

        a_m = self.input["a1_m"]
        b_m = self.input["b1_m"]

        z_m1 = a_m | b_m
        z_m2 = Fiber.union(a_m, b_m)

        for test, z_m in enumerate([z_m1, z_m2]):
            with self.subTest(test=test):
                # Check for right answer
                self.assertEqual(z_m, ans)

                # Check that payloads are of correct type
                self.assertIsInstance(z_m[0].payload.value[1], Payload)
                self.assertIsInstance(z_m[2].payload.value[1], Payload)
                self.assertIsInstance(z_m[3].payload.value[2], Payload)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, 0)))
                self.assertIsInstance(z_m_default, Payload)

                # Check final shape is correct
                z_m_shape = z_m.getShape()
                self.assertEqual(z_m_shape, [7])


    def test_union_2x_2d(self):
        """Test union 2-way for 2d fibers"""

        ans = Fiber([0, 1, 2, 4],
                    [('AB',
                      Fiber([0, 2, 4, 6], [1, 3, 5, 7]),
                      Fiber([0, 2, 3], [2, 4, 5])),
                     ('A',
                      Fiber([0, 1, 3, 6], [2, 2, 3, 8]),
                      Fiber([], [])),
                     ('B',
                      Fiber([], []),
                      Fiber([0, 1, 2], [3, 4, 6])),
                     ('AB',
                      Fiber([0, 2, 4, 6], [4, 5, 8, 9]),
                      Fiber([0, 1, 2, 3], [1, 2, 3, 4]))])

        a_m = self.input["a2_m"]
        b_m = self.input["b2_m"]

        z_m1 = a_m | b_m
        z_m2 = Fiber.union(a_m, b_m)

        for test, z_m in enumerate([z_m1, z_m2]):
            with self.subTest(test=test):
                # Check for right answer
                self.assertEqual(z_m, ans)

                # Check that payloads are of correct type
                self.assertIsInstance(z_m[0].payload.value[1], Fiber)
                self.assertIsInstance(z_m[0].payload.value[2], Fiber)
                self.assertIsInstance(z_m[2].payload.value[1], Fiber)
                self.assertIsInstance(z_m[3].payload.value[2], Fiber)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', Fiber, Fiber)))
                self.assertIsInstance(z_m_default, Payload)

                # Check final shape is correct (note it is 1-D)
                z_m_shape = z_m.getShape()
                self.assertEqual(z_m_shape, [5])


    def test_union_2x_1d2d(self):
        """Test union 2-way for 1d/2d fibers"""

        ans = Fiber([0, 2, 4, 6],
                    [('AB', 1, Fiber([0, 2, 3], [2, 4, 5])),
                     ('AB', 3, Fiber([0, 1, 2], [3, 4, 6])),
                     ('AB', 5, Fiber([0, 1, 2, 3], [1, 2, 3, 4])),
                     ('A', 7, Fiber([], []))])


        a_m = self.input["a1_m"]
        b_m = self.input["b2_m"]

        z_m1 = a_m | b_m
        z_m2 = Fiber.union(a_m, b_m)

        for test, z_m in enumerate([z_m1, z_m2]):
            with self.subTest(test=test):
                # Check for right answer
                self.assertEqual(z_m, ans)
                
                # Check that payloads are of correct type
                self.assertIsInstance(z_m[0].payload.value[1], Payload)
                self.assertIsInstance(z_m[0].payload.value[2], Fiber)
                self.assertIsInstance(z_m[2].payload.value[1], Payload)
                self.assertIsInstance(z_m[3].payload.value[2], Fiber)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, Fiber)))
                self.assertIsInstance(z_m_default, Payload)

                # Check final shape is correct (note it is 1-D)
                z_m_shape = z_m.getShape()
                self.assertEqual(z_m_shape, [7])


    def test_union_3x_1d(self):
        """Test union 3-way for 1d fibers"""

        ans = Fiber([0, 1, 2, 3, 4, 6],
                    [('ABC', Payload(1), Payload(2), Payload(1)),
                     ('C', Payload(0), Payload(0), Payload(2)),
                     ('ABC', Payload(3), Payload(4), Payload(3)),
                     ('B', Payload(0), Payload(5), Payload(0)),
                     ('A', Payload(5), Payload(0), Payload(0)),
                     ('A', Payload(7), Payload(0), Payload(0))])

        a_m = self.input["a1_m"]
        b_m = self.input["b1_m"]
        c_m = self.input["c1_m"]

        z_m1 = Fiber.union(a_m, b_m, c_m)

        for test, z_m in enumerate([z_m1]):
            with self.subTest(test=test):
                # Check for right answer
                self.assertEqual(z_m, ans)

                # Check that payloads are of correct type
                self.assertIsInstance(z_m[0].payload.value[1], Payload)
                self.assertIsInstance(z_m[0].payload.value[2], Payload)
                self.assertIsInstance(z_m[0].payload.value[3], Payload)
                self.assertIsInstance(z_m[1].payload.value[1], Payload)
                self.assertIsInstance(z_m[1].payload.value[2], Payload)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, 0, 0)))
                self.assertIsInstance(z_m_default, Payload)

                # Check final shape is correct (note it is 1-D)
                z_m_shape = z_m.getShape()
                self.assertEqual(z_m_shape, [7])
        

if __name__ == '__main__':
    unittest.main()

