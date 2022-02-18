import unittest
from copy import deepcopy

from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor


class TestFiberTensorUpdateDeep(unittest.TestCase):

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


    def test_updateCoords_deep(self):
        """Test updateCoords deep - assumes basic updateCoords works"""

        f = self.input['f']
        t = self.input['t']

        rankids = t.getRankIds()

        c_updates = [lambda n, c, p: c+1,
                     lambda n, c, p: 20-c]

        for depth, rankid in enumerate(rankids):

            for n, update in enumerate(c_updates):
                test = f"tensor: updateCoords(lambda[{n}], [depth={depth} | rankid='{rankid}'])"
                with self.subTest(test=test):

                    t1 = t.updateCoords(update, depth=depth)
                    t2 = t.updateCoords(update, rankid=rankid)

                    self.assertEqual(t1, t2)

                    #
                    # Note: Fiber.updateCoords() mutates the fiber, so we create a copy
                    #
                    f1 = deepcopy(f)
                    f1.updateCoords(update, depth=depth)

                    self.assertEqual(f1, t2.getRoot())

                    f2 = deepcopy(f)
                    f2.updateCoords(update, rankid=rankid)

                    self.assertEqual(f2, f1)

    def test_updatePayloads_deep(self):
        """Test updatePayloads deep - assumes basic updatePayloads works"""

        f = self.input['f']
        t = self.input['t']

        depth = 2
        rankid = "W"

        update = lambda i, c, p: p+1

        t1 = t.updatePayloads(update, depth=depth)
        t2 = t.updatePayloads(update, rankid=rankid)

        self.assertEqual(t1, t2)

        #
        # Note: Fiber.updateCoords() mutates the fiber, so we create a copy
        #
        f1 = deepcopy(f)
        f1.updatePayloads(update, depth=depth)

        self.assertEqual(f1, t2.getRoot())

        f2 = deepcopy(f)
        f2.updatePayloads(update, rankid=rankid)

        self.assertEqual(f2, f1)


if __name__ == '__main__':
    unittest.main()

