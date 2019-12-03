import unittest
from fibertree.payload import Payload
from fibertree.fiber import Fiber


class TestFiber(unittest.TestCase):

    def test_new_1d(self):
        """Create a 1d fiber"""

        a = Fiber([2, 4, 6], [3, 5, 7])

    def test_new_2d(self):
        """Create a 1d fiber"""

        b0 = Fiber([1, 4, 7], [2, 5, 8])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        a0 = Fiber([2, 4], [b0, b1])

    def test_comparison(self):

        a = Fiber([2, 4, 6], [3, 5, 7])
        b = Fiber([2, 4, 6], [3, 5, 7])

        self.assertEqual(a, b)

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


    def test_values(self):
        """Count values in a 2-D fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(a.values(), 6)


    def test_len(self):
        """Find lenght of a fiber"""

        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        self.assertEqual(len(a), 2)



        
    def test_payload(self):
        """Access payloads"""

        coords = [2, 4, 6]
        payloads = [3, 5, 7]

        a = Fiber(coords, payloads) 

        test = [0, 4, 6, 3]
        answer = [None, 5, 7, None]
        
        for i in range(len(test)):
            self.assertTrue(a.payload(test[i]) == answer[i])

    def test_insert(self):
        """"Insert payload at coordinates 0, 3, 7"""

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

        uncompressed_ref = [ [ 0, 0, 0, 0, 0, 0, 0, 0,],
                             [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                             [ 0, 2, 0, 0, 5, 0, 0, 8 ],
                             [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                             [ 0, 0, 3, 0, 5, 0, 7, 0 ] ]
                               
                          
        a = Fiber.fromYAMLfile("./data/test_fiber-2.yaml")

        uncompressed = a.uncompress()

        self.assertEqual(uncompressed, uncompressed_ref)

    
"""



    print("Iterator Print")
    i = a.__iter__()

    while True:
        try:
            coord, payload = next(i)
            print("(%s, %s)" % (coord, payload))
        except StopIteration:
            print("End")
            break
    print("----\n\n")


    print("Intersection")

    b = Fiber([2, 6, 8], [4, 8, 10])

    a.print()
    b.print()

    ab = a & b
    ab.print()
    print("----\n\n")

    print("For Intersection")

    for coord, payload in ab:
        print("(%s, %s)" % (coord, payload))

    print("----\n\n")

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

