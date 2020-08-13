
class CoordPayload():

    def __init__(self, coord, payload):
        """__init__"""

        self.coord = coord
        self.payload = payload

    def __iter__(self):
        """__iter__"""

        yield self.coord
        yield self.payload

    def __add__(self, other):
        """__add__"""

        if isinstance(other, CoordPayload):
            ans = self.payload + other.payload
        else:
            ans = self.payload + other

        return ans

    def __radd__(self, other):
        """__radd__"""

        return other + self.payload

    def __iadd__(self, other):
        """__iadd__"""

        if isinstance(other, CoordPayload):
            self.payload += other.payload
        else:
            self.payload += other

        return self

    def __sub__(self, other):
        """__sub__"""

        if isinstance(other, CoordPayload):
            ans = self.payload - other.payload
        else:
            ans = self.payload - other

        return ans

    def __rsub__(self, other):
        """__rsub__"""

        return other - self.payload

    def __isub__(self, other):
        """__isub__"""

        if isinstance(other, CoordPayload):
            self.payload -= other.payload
        else:
            self.payload -= other

        return self

    def __mul__(self, other):
        """__mul__"""

        if isinstance(other, CoordPayload):
            ans = self.payload * other.payload
        else:
            ans = self.payload * other

        return ans

    def __rmul__(self, other):
        """__rmul__"""

        return other * self.payload

    def __imul__(self, other):
        """__imul__"""

        if isinstance(other, CoordPayload):
            self.payload *= other.payload
        else:
            self.payload *= other

        return self

    def __div__(self, other):
        """__div__"""

        if isinstance(other, CoordPayload):
            ans = self.payload / other.payload
        else:
            ans = self.payload / other

        return ans

    def __rdiv__(self, other):
        """__rdiv__"""

        return other / self.payload

    def __idiv__(self, other):
        """__idiv__"""

        if isinstance(other, CoordPayload):
            self.payload /= other.payload
        else:
            self.payload /= other

        return self

    def __repr__(self):
        """__repr__"""

        return str(f"CoordPayload(coord={self.coord}, payload={self.payload})")


#
# Comparison operations
#

    def __eq__(self, other):
        """__eq__"""

        if isinstance(other, CoordPayload):
            return self.payload == other.payload

        return self.payload == other

    def __lt__(self, other):
        """__lt__"""

        if isinstance(other, CoordPayload):
            return self.payload < other.payload

        return self.payload < other

    def __le__(self, other):
        """__le__"""

        if isinstance(other, CoordPayload):
            return self.payload <= other.payload

        return self.payload <= other

    def __gt__(self, other):
        """__gt__"""

        if isinstance(other, CoordPayload):
            return self.payload > other.payload

        return self.payload > other

    def __ge__(self, other):
        """__ge__"""

        if isinstance(other, CoordPayload):
            return self.payload >= other.payload

        return self.payload >= other

    def __ne__(self, other):
        """__ne__"""

        if isinstance(other, CoordPayload):
            return self.payload != other.payload

        return self.payload != other


##############################################

if __name__ == "__main__":

    from fibertree import Payload

    print("")
    print("Start test")
    print("")

    a = CoordPayload(5, 4)
    print(f"a = CoordPayload(5, 4) -> {a}")

    x, y = a
    print(x, y)

    print("")

    b = CoordPayload(coord=6, payload=2)
    print(f"b = CoordPayload(6, 2) -> {b}")

    print("")

    z = a + b
    print(f"a+b -> {z}")

    z = a + 1
    print(f"a+1 -> {z}")

    z = 1 + a
    print(f"1+a -> {z}")

    print("")

    p = Payload(4)
    print(f"p = Payload(4) -> {p}")

    z = a + p
    print(f"a+p -> {z}")

    z = p + a
    print(f"p+a -> {z}")

    print("")

    a += b
    print(f"a+=b -> {a}")
    a = CoordPayload(5, 4)

    a += 2
    print(f"a+=2 -> {a}")
    a = CoordPayload(5, 4)

    a += p
    print(f"a+=p -> {a}")
    a = CoordPayload(5, 4)

    print("")

    c = CoordPayload(coord=7, payload=Payload(8))
    print(f"c = CoordPayload(7, Payload(8)) -> {c}")

    print("")

    z = a + c
    print(f"a+c -> {z}")

    z = c + a
    print(f"c+a -> {z}")

    z = c + 1
    print(f"c+1 -> {z}")

    z = 1 + c
    print(f"1+c -> {z}")

    print("")

    c += b
    print(f"c+=b -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))

    c += 2
    print(f"c+=2 -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))

    c += p
    print(f"c+=p -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))

    print("")

    a = CoordPayload(5, 4)
    print(f"a = CoordPayload(5, 4) -> {a}")

    x, y = a
    print(x, y)

    print("")

    b = CoordPayload(coord=6, payload=2)
    print(f"b = CoordPayload(6, 2) -> {b}")

    print("")

    z = a * b
    print(f"a*b -> {z}")

    z = a * 1
    print(f"a*1 -> {z}")

    z = 1 * a
    print(f"1*a -> {z}")

    print("")

    p = Payload(4)
    print(f"p = Payload(4) -> {p}")

    z = a * p
    print(f"a*p -> {z}")

    z = p * a
    print(f"p*a -> {z}")

    print("")

    a *= b
    print(f"a*=b -> {a}")
    a = CoordPayload(5, 4)

    a *= 2
    print(f"a*=2 -> {a}")
    a = CoordPayload(5, 4)

    a *= p
    print(f"a*=p -> {a}")
    a = CoordPayload(5, 4)

    print("")

    c = CoordPayload(coord=7, payload=Payload(8))
    print(f"c = CoordPayload(7, Payload(8)) -> {c}")

    print("")

    z = a * c
    print(f"a*c -> {z}")

    z = c * a
    print(f"c*a -> {z}")

    z = c * 1
    print(f"c*1 -> {z}")

    z = 1 * c
    print(f"1*c -> {z}")

    print("")

    c *= b
    print(f"c*=b -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))

    c *= 2
    print(f"c*=2 -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))

    c *= p
    print(f"c*=p -> {c}")
    c = CoordPayload(coord=7, payload=Payload(8))
