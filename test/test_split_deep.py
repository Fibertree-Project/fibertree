import unittest
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor


class TestFiberTensorSplitDeep(unittest.TestCase):

    def setUp(self):

        self.input = {}

        self.input['f'] = Fiber([0, 1, 2],
                                [Fiber([0, 1, 2],
                                       [Fiber([0, 1, 2, 3], [2, 4, 2, 4]),
                                        Fiber([0, 1, 2, 3], [3, 2, 4, 3]),
                                        Fiber([0, 1, 3], [3, 2, 1])]),
                                 Fiber([0, 1, 2],
                                       [Fiber([0, 1, 2, 3], [3, 2, 5, 4]),
                                        Fiber([0, 1, 3], [5, 2, 1]),
                                        Fiber([0, 1, 2], [1, 1, 5])]),
                                 Fiber([0, 1, 2],
                                       [Fiber([1, 2], [4, 2]),
                                        Fiber([0, 1, 2, 3], [2, 2, 2, 3]),
                                        Fiber([0, 1, 2, 3], [2, 4, 3, 1])])])

        self.input['t'] = Tensor.fromFiber(["C", "H", "W"], self.input['f'])


    def test_split_deep(self):
        """Test splitDeep - assumes basic split works"""

        f = self.input['f']
        t = self.input['t']

        rankids = t.getRankIds()

        for depth, rankid in enumerate(rankids):

            with self.subTest(test=f"splitUniform(2, [depth={depth} | rankid='{rankid}'])"):
                t1a = t.splitUniform(2, depth=depth)
                t1b = t.splitUniform(2, rankid=rankid)
                self.assertEqual(t1b,t1a)

                i1a = f.splitUniform(2, depth=depth)
                self.assertEqual(i1a, t1b.getRoot())

                i1b = f.splitUniform(2, rankid=rankid)
                self.assertEqual(i1b, t1a.getRoot())


            with self.subTest(test=f"splitNonUniform[0,1], [depth={depth} | rankid='{rankid}'])"):

                t2a = t.splitNonUniform([0, 1], depth=depth)
                t2b = t.splitNonUniform([0, 1], rankid=rankid)
                self.assertEqual(t2b, t2a)

                i2a = f.splitNonUniform([0, 1], depth=depth)
                self.assertEqual(i2a, t2b.getRoot())

                i2b = f.splitNonUniform([0, 1], rankid=rankid)
                self.assertEqual(i2b, i2a)

            with self.subTest(test=f"splitEqual(2, [depth={depth} | rankid='{rankid}'])"):

                t3a = t.splitEqual(2, depth=depth)
                t3b = t.splitEqual(2, rankid=rankid)
                self.assertEqual(t3b, t3a)

                i3a = f.splitEqual(2, depth=depth)
                self.assertEqual(i3a, t3b.getRoot())

                i3b = f.splitEqual(2, rankid=rankid)
                self.assertEqual(i3b, i3a)


            with self.subTest(test=f"splitUnEqual([1,3], [depth={depth}, rankid='{rankid}'])"):

                 t4a = t.splitUnEqual([1,3], depth=depth)
                 t4b = t.splitUnEqual([1,3], rankid=rankid)
                 self.assertEqual(t4b, t4a)

                 i4a = f.splitUnEqual([1,3], depth=depth)
                 self.assertEqual(i4a, t4b.getRoot())

                 i4b = f.splitUnEqual([1,3], rankid=rankid)
                 self.assertEqual(i4b, i4a)



if __name__ == '__main__':
    unittest.main()

