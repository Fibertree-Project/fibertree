import unittest

from fibertree.payload import Payload
from fibertree.fiber import Fiber
from fibertree.rank import Rank
from fibertree.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_new(self):
        tensor = Tensor("./data/test_tensor-1.yaml")
        tensor.print()
        pass


if __name__ == '__main__':
    unittest.main()

