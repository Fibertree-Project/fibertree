import unittest
from fibertree.payload import Payload
from fibertree.fiber import Fiber


class TestFiber(unittest.TestCase):

    def test_new(self):
        """Create a fiber"""

        a = Fiber([2, 4, 6], [3, 5, 7])

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

            a.print("List with (%s, %s) inserted" % (i, p))
            ans[i].print()

            x = a == ans[i]
            print(x)

#            self.assertTrue(a == ans[i])


    
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

