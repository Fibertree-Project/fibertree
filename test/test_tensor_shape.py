"""Tests related to shape of a tensor"""

import unittest

from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.rank import Rank
from fibertree.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_shape_empty(self):
        """Test shape of empty tensor"""

        t1 = Tensor(rank_ids=["M", "K"])

        self.assertEqual(t1.getRankIds(), ["M", "K"])
        self.assertEqual(t1.getShape(), [0, 0])

        t2 = Tensor(rank_ids=["M", "K"], shape=[10,20])

        self.assertEqual(t2.getRankIds(), ["M", "K"])
        self.assertEqual(t2.getShape(), [10, 20])

    def test_shape_0D(self):
        """Test shpe of 0-D tensor"""

        t = Tensor(rank_ids=[])
        p = t.getRoot()
        p += 1

        self.assertEqual(t.getRankIds(), [])
        self.assertEqual(t.getShape(), [])
        
    def test_shape_new(self):
        """Test shape of a tensor from a file"""

        t1 = Tensor("./data/test_tensor-1.yaml")

        self.assertEqual(t1.getRankIds(), ["M", "K"])
        self.assertEqual(t1.getShape(), [7, 4])


        # Note: We cannot override the shape of shape from a YAML file


    def test_shape_fromUncompressed_1D(self):
        """Test shape of a tensor from 1D nested lists"""

        l1 = [ 100, 101, 0, 102 ]

        t1 = Tensor.fromUncompressed(["M"], l1)

        self.assertEqual(t1.getRankIds(), ["M"])
        self.assertEqual(t1.getShape(), [ 4 ])

        l2 = [ 100, 101, 0, 0 ]

        t2 = Tensor.fromUncompressed(["M"], l2)

        self.assertEqual(t2.getRankIds(), ["M"])
        self.assertEqual(t2.getShape(), [ 4 ])

    def test_shape_fromUncompressed_2D(self):
        """Test shape of a tensor from 2D nested lists"""

        #         0    1    2    3
        #
        l1 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0, 203 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0, 603 ] ] # 6

        t1 = Tensor.fromUncompressed(["M", "K"], l1)

        self.assertEqual(t1.getRankIds(), ["M", "K"])
        self.assertEqual(t1.getShape(), [ 7, 4 ])

        #         0    1    2    3
        #
        l2 = [ [   0,   0,   0,   0 ],  # 0
               [ 100, 101, 102,   0 ],  # 1
               [   0, 201,   0,   0 ],  # 2
               [   0,   0,   0,   0 ],  # 3
               [ 400,   0, 402,   0 ],  # 4
               [   0,   0,   0,   0 ],  # 5
               [   0, 601,   0,   0 ] ] # 6

        t2 = Tensor.fromUncompressed(["M", "K"], l2)

        self.assertEqual(t2.getRankIds(), ["M", "K"])
        self.assertEqual(t2.getShape(), [ 7, 4 ])


    def test_shape_fromFiber(self):
        """Test shape of a tensor from a fiber"""

        y1 = Tensor("./data/test_tensor-1.yaml")
        f1 = y1.getRoot()

        t1 = Tensor.fromFiber(["K", "M"], f1)
        
        self.assertEqual(t1.getRankIds(), ["K", "M"])
        self.assertEqual(t1.getShape(), [7, 4])
    

        y2 = Tensor("./data/test_tensor-1.yaml")
        f2 = y2.getRoot()
        t2 = Tensor.fromFiber(["K100", "M100"], f2, [100,200] )
        
        self.assertEqual(t2.getRankIds(), ["K100", "M100"])
        self.assertEqual(t2.getShape(), [100, 200])
    
        
if __name__ == '__main__':
    unittest.main()

