import unittest

from fibertree import Metrics
from fibertree import Payload

class TestPayload(unittest.TestCase):

    def test_new(self):
        av = 1
        bv = 1

        a = Payload(av)
        self.assertEqual(a.value, av)

        b = Payload(bv)
        self.assertEqual(b.value, bv)

    def test_plus(self):
        av = 1
        bv = 1

        a = Payload(av)
        b = Payload(bv)

        a_plus_1 = a + 1
        self.assertEqual(a_plus_1.value, av+1)

        a_plus_b = a + b
        self.assertEqual(a_plus_b.value, av+bv)

        a += 1
        self.assertEqual(a.value, av+1)

        a += b
        self.assertEqual(a.value, av+1+bv)

    def test_plus_metrics(self):
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)

        Metrics.beginCollect()
        _ = a + b
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_add": 1}})

        _ = a + 2
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_add": 2}})

        _ = 1 + b
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_add": 3}})

        a += b
        self.assertEqual(Metrics.dump(), {
            "Compute": {"payload_add": 4, "payload_update": 1}})

        Metrics.endCollect()

    def test_minus(self):
        av = 1
        bv = 1

        a = Payload(av)
        b = Payload(bv)

        a_sub_1 = a - 1
        self.assertEqual(a_sub_1.value, av-1)

        a_sub_b = a - b
        self.assertEqual(a_sub_b.value, av-bv)

    def test_multiply(self):
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)


        a_mul_2 = a * 2
        self.assertEqual(a_mul_2.value, av*2)

        two_mul_a = 2 * a
        self.assertEqual(two_mul_a.value, 2*av)

        a_mul_b = a * b
        self.assertEqual(a_mul_b.value, av*bv)

        a *= 2
        self.assertEqual(a.value, av*2)

        a *= b
        self.assertEqual(a.value, av*2*b)

    def test_mul_metrics(self):
        av = 1
        bv = 2

        a = Payload(av)
        b = Payload(bv)

        Metrics.beginCollect()
        _ = a * b
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_mul": 1}})

        _ = a * 2
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_mul": 2}})

        _ = 1 * b
        self.assertEqual(Metrics.dump(), {"Compute": {"payload_mul": 3}})

        a *= b
        self.assertEqual(Metrics.dump(), {
            "Compute": {"payload_mul": 4, "payload_update": 1}})

        Metrics.endCollect()

    def test_equality(self):
        cv = 8
        dv = 8
        ev = 1

        c = Payload(cv)
        d = Payload(dv)
        e = Payload(ev)

        self.assertTrue(c == d)
        self.assertFalse(c == e)

        self.assertFalse(c != d)
        self.assertTrue(c != e)

        self.assertTrue(c == 8)
        self.assertFalse(c == 1)
        self.assertFalse(c != 8)
        self.assertTrue(c != 1)

        self.assertTrue(8 == c)
        self.assertFalse(1 == c)
        self.assertFalse(8 != c)
        self.assertTrue(1 != c)



if __name__ == '__main__':
    unittest.main()

