import unittest
from fibertree import Payload
from fibertree import Fiber


class TestFiberInfixSplit(unittest.TestCase):

    def setUp(self):

        self.input = {}

        self.input[0] = Fiber([3, 6, 8, 9, 12, 16, 19, 20, 28, 30, 32, 38, 40, 43, 46, 47, 48, 49],
                              [8, 9, 6, 3, 5, 4, 1, 4, 6, 4, 1, 6, 2, 6, 5, 9, 2, 5])


    def test_split_uniform(self):
        """Test splitUniform"""

        self.ans = {}

        self.ans[2] = Fiber([0, 25],
                            [Fiber([3, 6, 8, 9, 12, 16, 19, 20],
                                   [8, 9, 6, 3, 5, 4, 1, 4]),
                             Fiber([28, 30, 32, 38, 40, 43, 46, 47, 48, 49],
                                   [6, 4, 1, 6, 2, 6, 5, 9, 2, 5])])

        self.ans[3] = Fiber([0, 17, 34],
                            [Fiber([3, 6, 8, 9, 12, 16],
                                   [8, 9, 6, 3, 5, 4]),
                             Fiber([19, 20, 28, 30, 32],
                                   [1, 4, 6, 4, 1]),
                             Fiber([38, 40, 43, 46, 47, 48, 49],
                                   [6, 2, 6, 5, 9, 2, 5])])

        self.ans[4] = Fiber([0, 13, 26, 39],
                            [Fiber([3, 6, 8, 9, 12],
                                   [8, 9, 6, 3, 5]),
                             Fiber([16, 19, 20],
                                   [4, 1, 4]),
                             Fiber([28, 30, 32, 38],
                                   [6, 4, 1, 6]),
                             Fiber([40, 43, 46, 47, 48, 49],
                                   [2, 6, 5, 9, 2, 5])])

        self.ans[5] = Fiber([0, 10, 20, 30, 40],
                            [Fiber([3, 6, 8, 9],
                                   [8, 9, 6, 3]),
                             Fiber([12, 16, 19],
                                   [5, 4, 1]),
                             Fiber([20, 28], [4, 6]),
                             Fiber([30, 32, 38],
                                   [4, 1, 6]),
                             Fiber([40, 43, 46, 47, 48, 49],
                                   [2, 6, 5, 9, 2, 5])])

        f = self.input[0]

        for p in range(2,6):
            with self.subTest(test=p):
                ans = f / p
                self.assertEqual(ans, self.ans[p])

    def test_split_uniform_eager_only(self):
        """Test that the uniform split allows eager mode only"""
        f = self.input[0]
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f / 2

        with self.assertRaises(AssertionError):
            f.splitUniform(2)

        with self.assertRaises(AssertionError):
            f.splitNonUniform([0, 10, 20, 30, 40])

    def test_split_equal(self):
        """Test splitEqual"""

        self.ans = {}

        self.ans[2] = Fiber([3, 30],
                            [Fiber([3, 6, 8, 9, 12, 16, 19, 20, 28],
                                   [8, 9, 6, 3, 5, 4, 1, 4, 6]),
                             Fiber([30, 32, 38, 40, 43, 46, 47, 48, 49],
                                   [4, 1, 6, 2, 6, 5, 9, 2, 5])])

        self.ans[3] = Fiber([3, 19, 40],
                            [Fiber([3, 6, 8, 9, 12, 16],
                                   [8, 9, 6, 3, 5, 4]),
                             Fiber([19, 20, 28, 30, 32, 38],
                                   [1, 4, 6, 4, 1, 6]),
                             Fiber([40, 43, 46, 47, 48, 49],
                                   [2, 6, 5, 9, 2, 5])])

        self.ans[4] = Fiber([3, 16, 32, 47],
                            [Fiber([3, 6, 8, 9, 12],
                                   [8, 9, 6, 3, 5]),
                             Fiber([16, 19, 20, 28, 30],
                                   [4, 1, 4, 6, 4]),
                             Fiber([32, 38, 40, 43, 46],
                                   [1, 6, 2, 6, 5]),
                             Fiber([47, 48, 49],
                                   [9, 2, 5])])

        self.ans[5] = Fiber([3, 12, 28, 40, 48],
                            [Fiber([3, 6, 8, 9],
                                   [8, 9, 6, 3]),
                             Fiber([12, 16, 19, 20],
                                   [5, 4, 1, 4]),
                             Fiber([28, 30, 32, 38],
                                   [6, 4, 1, 6]),
                             Fiber([40, 43, 46, 47],
                                   [2, 6, 5, 9]),
                             Fiber([48, 49],
                                   [2, 5])])
        f = self.input[0]

        for p in range(2,6):
            with self.subTest(test=p):
                ans = f // p
                self.assertEqual(ans, self.ans[p])

    def test_split_equal_eager_only(self):
        """Test that the equal split allows eager mode only"""
        f = self.input[0]
        f._setIsLazy(True)

        with self.assertRaises(AssertionError):
            f // 2

        with self.assertRaises(AssertionError):
            f.splitEqual(2)

        with self.assertRaises(AssertionError):
            f.splitUnEqual([4, 3, 2, 3, 6])

if __name__ == '__main__':
    unittest.main()

