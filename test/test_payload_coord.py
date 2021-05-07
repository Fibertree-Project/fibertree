import unittest

from fibertree import Fiber, Payload,  CoordPayload

class TestPayloadCoord(unittest.TestCase):

    def test_new(self):

        cp = CoordPayload(1, 10)

        self.assertEqual(cp.coord, 1)
        self.assertEqual(cp.payload, 10)

    def test_iter(self):

        cp_ref = [1, 10]

        cp = CoordPayload(1, 10)

        for x, x_ref in zip(cp, cp_ref):
            self.assertEqual(x, x_ref)

    def test_getitem_2D(self):

        b0 = Fiber([1, 4, 7], [2, 5, 8])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        a0 = Fiber([2, 4], [b0, b1])

        self.assertEqual(a0[1][1], 5)
        

    def test_setitem_2D(self):

        b0 = Fiber([1, 4, 7], [2, 5, 8])
        b1 = Fiber([2, 4, 6], [3, 5, 7])
        a0 = Fiber([2, 4], [b0, b1])

        a0[1][1] = 10

        self.assertEqual(a0[1][1], 10)


    def test_add(self):

        cp1 = CoordPayload(5, 11)
        cp2 = CoordPayload(6, 12)

        payload_ref = Payload(23)
        
        self.assertEqual(cp1+12, payload_ref)
        self.assertEqual(cp1+cp2, payload_ref)

        self.assertEqual(12+cp1, payload_ref)

        cp1 += cp2
        self.assertEqual(cp1, payload_ref)

        cp2 += 11
        self.assertEqual(cp2, payload_ref)

    def test_sub(self):

        cp1 = CoordPayload(5, 12)
        cp2 = CoordPayload(6, 10)

        payload_ref = Payload(2)
        
        self.assertEqual(cp1-10, payload_ref)
        self.assertEqual(cp1-cp2, payload_ref)

        self.assertEqual(14-cp1, payload_ref)

        cp1 -= cp2
        self.assertEqual(cp1, payload_ref)

        cp2 -= 8
        self.assertEqual(cp2, payload_ref)
        
    def test_multiply(self):

        cp1 = CoordPayload(5, 4)
        cp2 = CoordPayload(6, 5)

        payload_ref = Payload(20)
        
        self.assertEqual(cp1*5, payload_ref)
        self.assertEqual(cp1*cp2, payload_ref)

        self.assertEqual(5*cp1, payload_ref)

        cp1 *= cp2
        self.assertEqual(cp1, payload_ref)

        cp2 *= 4
        self.assertEqual(cp2, payload_ref)

#
#    Not working
#
#    def test_div(self):
#
#        cp1 = CoordPayload(5, 20)
#        cp2 = CoordPayload(6, 4)
#
#        payload_ref = Payload(5)
#        
#        self.assertEqual(cp1/4.0, payload_ref)
#        self.assertEqual(cp1/cp2, payload_ref)
#
#        self.assertEqual(20/cp2, payload_ref)
#
#       cp1 /= cp2
#        self.assertEqual(cp1, payload_ref)
#
#        cp2 /= 4
#        self.assertEqual(cp2, Payload(1))

    def test_comparisons(self):

        cp1 = CoordPayload(5, 4)
        cp2 = CoordPayload(6, 5)
        cp1a = CoordPayload(5, 4)

        self.assertTrue(cp1 == cp1a)
        self.assertTrue(cp1 == 4)

        self.assertFalse(cp1 == cp2)
        self.assertFalse(cp1 == 5)
        
        self.assertTrue(cp1 < cp2)
        self.assertTrue(cp1 < 5)

        self.assertFalse(cp2 < cp1)
        self.assertFalse(cp2 < 5)

        self.assertTrue(cp1 <= cp1a)
        self.assertTrue(cp1 <= cp2)
        self.assertTrue(cp1 <= 4)
        self.assertTrue(cp1 <= 5)        

        self.assertTrue(cp2 > cp1)
        self.assertTrue(cp2 > 4)

        self.assertTrue(cp1 >= cp1a)
        self.assertTrue(cp2 >= cp1)
        self.assertTrue(cp1 >= 4)
        self.assertTrue(cp1 >= 3)        

        self.assertTrue(cp1 != cp2)
        self.assertTrue(cp1 != 5)

        self.assertFalse(cp1 != cp1)
        self.assertFalse(cp1 != 4)
