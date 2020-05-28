""" Payload class"""

class Payload:
    """Payload class"""

    def __init__(self, value=None):
        """__init__"""

        self.value = value

    def v(self):
        """v"""

        return self.value

    def __setattr__(self, name, value):
        """__setattr__"""

        if name == "v":
            name = "value"

        # If value is a Payload copy in its value
        if isinstance(value, Payload):
            value = value.v()

        self.__dict__[name] = value

    def __iter__(self):
        """__iter__"""

        for v in self.value:
            yield v

    def __reversed__(self):
        """__reversed__"""

        return reversed(self.value)


    def __bool__(self):
        """__bool__"""

        return bool(self.value)

    def __int__(self):
        """__int__"""

        return int(self.value)

#
# Static methods
#
    @staticmethod
    def isEmpty(p):

        from fibertree.fiber import Fiber

        if isinstance(p, Fiber):
            return p.isEmpty()

        if isinstance(p, tuple):
            assert isinstance(p, tuple)

        if (p == 0):
            return  True

        return False

#
#
# Transition methods
#
# Note: the following two methods are used as part of the transition from
#       Fibers holding a raw Fiber as a payload to that Fiber being embedded
#       in in a Payload object
#
    @staticmethod
    def contains(payload, type):
        """Return whether "payload" is of type "type" - checking inside payload if necessary"""

        if not isinstance(payload, Payload):
            return isinstance(payload, type)

        return isinstance(payload.value, type)

    @staticmethod
    def get(payload):
        """Return value of "payload" - checking inside payload if necessary"""

        if not isinstance(payload, Payload):
            return payload

        return payload.value

#
# Srtring operations
#

    def print(self, title=None):
        """print"""

        return self.value.print(title)


    def __format__(self, spec="d"):
        """__format__"""

        return f"<{self.value:{spec}}>"


    def __str__(self):
        """__str__"""

        return "<%s>" % self.value.__str__()

    def __repr__(self):
        """__repr__"""

        return "%s" % self.value

#
# Arithmetic operations
#
    def __add__(self, other):
        """__add__"""

        if isinstance(other, Payload):
            ans = self.value + other.value
        else:
            ans = self.value + other

        return Payload(ans)

    def __radd__(self, other):
        """__radd__"""

        assert not isinstance(other, Payload)

        return Payload(other + self.value)

    def __iadd__(self, other):
        """__iadd__"""

        if isinstance(other, Payload):
            self.value = self.value + other.value
        else:
            self.value = self.value + other
        return self

    def __sub__(self, other):
        """__sub__"""

        if isinstance(other, Payload):
            ans = self.value - other.value
        else:
            ans = self.value - other

        return Payload(ans)

    def __rsub__(self, other):
        """__rsub__"""

        assert not isinstance(other, Payload)
        return Payload(other - self.value)


    def __isub__(self, other):
        """__isub__"""

        if isinstance(other, Payload):
            self.value = self.value - other.value
        else:
            self.value = self.value - other
        return self


    def __mul__(self, other):
        """__mul__"""

        if isinstance(other, Payload):
            ans = self.value * other.value
        else:
            ans = self.value * other

        return Payload(ans)

    def __truediv__(self, other):
        """__truediv__"""

        if isinstance(other, Payload):
            ans = self.value / other.value
        else:
            ans = self.value / other

        return Payload(ans)

    def __rmul__(self, other):
        """__rmul__"""

        assert not isinstance(other, Payload)

        return Payload(other * self.value)


    def __imul__(self, other):
        """__imul__"""

        if isinstance(other, Payload):
            self.value = self.value * other.value
        else:
            self.value = self.value * other
        return self


#
# Comparison operations
#

    def __eq__(self, other):
        """__eq__"""

        if isinstance(other, Payload):
            return self.value == other.value

        return self.value == other

    def __ne__(self, other):
        """__ne__"""

        if isinstance(other, Payload):
            return self.value != other.value

        return self.value != other

#
# Logical operatons
#    Note: primarily used by fiber iterators
#

    def __and__(self, other):
        """__and__"""

        if isinstance(other, Payload):
            ans = self.value & other.value
        else:
            ans = self.value & other

        return Payload(ans)


    def __or__(self, other):
        """__or__"""

        if isinstance(other, Payload):
            ans = self.value | other.value
        else:
            ans = self.value | other

        return Payload(ans)


    def __lshift__(self, other):
        """__lshift__"""

        if isinstance(other, Payload):
            ans = self.value << other.value
        else:
            ans = self.value << other

        return Payload(ans)

#
# Conversion methods - to/from dictionaries
#

    @staticmethod
    def payload2dict(payload):
        """Return payload converted to dictionry or simple value"""

        from fibertree.fiber import Fiber

        if isinstance(payload, Fiber):
            # Note: this leg is deprecated and should be removed
            return payload.fiber2dict()
        elif isinstance(payload, Payload):
            if Payload.contains(payload, Fiber):
                return payload.value.fiber2dict()
            else:
                return payload.value
        else:
            return payload



if __name__ == "__main__":

    a = Payload(1)
    print("A = %s" % a)
    print("---")

