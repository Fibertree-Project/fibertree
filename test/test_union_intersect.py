import os
import unittest

from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor
from fibertree import Metrics


class TestUnionIntersect(unittest.TestCase):

    def setUp(self):
        # Make sure that no metrics are being collected, unless explicitly
        # desired by the test
        Metrics.endCollect()

        # Make sure we have a tmp directory to write to
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

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

    def test_intersection(self):
        """Test the intersection() function"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 5])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 6, 7, 0, 8])
        b_k.getRankAttrs().setId("K")
        c_k = Fiber.fromUncompressed([10, 0, 9, 0, 12])
        c_k.getRankAttrs().setId("K")

        # Check the elements
        cc = [2, 4]
        cp = [(3, 7, 9), (5, 8, 12)]

        ans = Fiber.intersection(a_k, b_k, c_k)
        for i, (c, p) in enumerate(ans):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

        # Check the fiber fields
        self.assertEqual(ans.getActive(), (0, 5))
        self.assertEqual(ans.getRankAttrs().getId(), "K")

    def test_intersection_metrics(self):
        """Test metrics collection on the intersection() function"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 4, 5])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 6, 7, 0, 8])
        b_k.getRankAttrs().setId("K")
        c_k = Fiber.fromUncompressed([10, 0, 9, 0, 12])
        c_k.getRankAttrs().setId("K")

        Metrics.beginCollect("tmp/test_intersection_metrics")
        Metrics.trace("K")
        Metrics.registerRank("M")
        for m in range(3):
            Metrics.addUse("M", m + 1, m)
            for _ in Fiber.intersection(a_k, b_k, c_k):
                pass
            Metrics.incIter("M")
        Metrics.endCollect()

        corr = [
            "M_pos,K_pos,M,K,fiber_pos\n",
            "0,3,1,2,0\n",
            "0,5,1,4,1\n",
            "1,3,2,2,0\n",
            "1,5,2,4,1\n",
            "2,3,3,2,0\n",
            "2,5,3,4,1\n"
        ]

        with open("tmp/test_intersection_metrics-K-iter.csv", "r") as f:
            self.assertEqual(f.readlines(), corr)

#     def test_intersection_tuple():
#         A_coords = [(0, 2), (0, 8), (2, 10), (3, 2), (3, 4), (3, 6), (4, 17)]
#         A_payloads = [1, 1, 2, 3, 5, 8, 13]
#
#         B_coords = [0, 1, 3]
#         B_payloads = [2, 7, 8]
#
#         a_km = Fiber(coords=A_coords, payloads=A_payloads)
#         b_k = Fiber(coords=B_coords, payloads=B_payloads)
#
#         class Any:
#             def __eq__(self, other):
#                 return True
#
#         corr = [(0, 2), (0, 8), (3, 2), (3, 4), (3, 6)]
#         for i, (km, (a_val, b_val)) in enumerate(a_km & b_k.project(trans_fn=lambda k: (k, Any()))):
#             self.assertEqual(km, corr[i])

    def test_union(self):
        """Test the union() function"""
        a_k = Fiber.fromUncompressed([1, 0, 3, 0, 0])
        a_k.getRankAttrs().setId("K")
        b_k = Fiber.fromUncompressed([0, 6, 7, 0, 8])
        b_k.getRankAttrs().setId("K")
        c_k = Fiber.fromUncompressed([10, 0, 9, 0, 12])
        c_k.getRankAttrs().setId("K")

        cc = [0, 1, 2, 4]
        cp = [("AC", 1, 0, 10), ("B", 0, 6, 0), ("ABC", 3, 7, 9), ("BC", 0, 8, 12)]

        ans = Fiber.union(a_k, b_k, c_k)
        for i, (c, p) in enumerate(ans):
            self.assertEqual(cc[i], c)
            self.assertEqual(cp[i], p)

        # Test fiber attributes
        self.assertEqual(ans.getRankAttrs().getId(), "K")
        self.assertEqual(ans.getActive(), (0, 5))

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
                for c, p in z_m:
                    # Check that the data is correct
                    self.assertEqual(ans.getPayload(c), p)

                    # Check that payloads are of correct type
                    _, a_val, b_val = p
                    self.assertIsInstance(a_val, Payload)
                    self.assertIsInstance(b_val, Payload)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, 0)))
                self.assertIsInstance(z_m_default, Payload)

                # Check the rank ID
                id_ = z_m.getRankAttrs().getId()
                self.assertEqual(id_, "M")

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
                for c, p in z_m:
                    # Check that the data is correct
                    self.assertEqual(ans.getPayload(c), p)

                    # Check that payloads are of correct type
                    _, a_val, b_val = p
                    self.assertIsInstance(a_val, Fiber)
                    self.assertIsInstance(b_val, Fiber)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', Fiber, Fiber)))
                self.assertIsInstance(z_m_default, Payload)

                # Check the rank ID
                id_ = z_m.getRankAttrs().getId()
                self.assertEqual(id_, "M")

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
                for c, p in z_m:
                    # Check that the data is correct
                    self.assertEqual(ans.getPayload(c), p)

                    # Check that payloads are of correct type
                    _, a_val, b_val = p
                    self.assertIsInstance(a_val, Payload)
                    self.assertIsInstance(b_val, Fiber)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, Fiber)))
                self.assertIsInstance(z_m_default, Payload)

                # Check the rank ID
                id_ = z_m.getRankAttrs().getId()
                self.assertEqual(id_, "M")

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
                for c, p in z_m:
                    # Check that the data is correct
                    self.assertEqual(ans.getPayload(c), p)

                    # Check that payloads are of correct type
                    _, a_val, b_val, c_val = p
                    self.assertIsInstance(a_val, Payload)
                    self.assertIsInstance(b_val, Payload)
                    self.assertIsInstance(c_val, Payload)

                # Check that default was set properly
                z_m_default=z_m.getDefault()
                self.assertEqual(z_m_default, Payload(('', 0, 0, 0)))
                self.assertIsInstance(z_m_default, Payload)

                # Check the rank ID
                id_ = z_m.getRankAttrs().getId()
                self.assertEqual(id_, "M")


if __name__ == '__main__':
    unittest.main()

