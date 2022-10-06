#cython: language_level=3
"""CoordPayload

A class used to represent an **element** of a fiber, i.e., a
coordinate/payload tuple.

"""
import logging
import pickle

from .payload import Payload

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.core.coord_payload')


class CoordPayload():
    """An element of a fiber.

    Instances of this class are returned by some `Fiber` methods, most
    significantly, iteration (see `Fiber.__iter__()`).

    In many instances one just wants to operate on the `payload` part
    of the element. Therefore, this class provides a variety of
    overloaded operations that do just that. These include index
    operations ([]) and many operators, including:

    - Addition: (+, +-)
    - Subtraction: (-, -=)
    - Multiplication: (*, *=)
    - Division: (/, /=)
    - Integer division: (//)
    - Left shift: (<<)
    - Boolean and: (&)
    - Boolean or: (|)
    - Equal: (=)
    - Not equal: (!=)
    - Less than: (<)
    - Great than: (&gt;)
    - Less than or equal: (<=)
    - Greater than or equal: (&gt;=)

    In addition to the above operators one needs to be able to
    conveniently assign a new value to the payload of a
    `CoordPayload`. Since the normal Python assignment operator (=)
    will replace a pointer to a class rather than update a value in
    the class (i.e., the **boxed** value) we overload the operator
    "<<=" to assign a new **boxed** value to the payload (see
    `CoordPayload.__ilshift__()`)

    Attributes
    ----------

    coord: a hashable value
        A value used as a coordinate

    payload: a legal payload
        A legal "payload" value.


    Note - these attributes are just left public in the class, so
    given an instance of this class named `element` one can access the
    attributes as `element.coord` and `element.payload`.

    Constructor
    -----------

    The `CoordPayload` constructor creates an element of a fiber with
    a given `coord` and `payload`.

    Parameters
    ----------

    coord: a hashable value
        A value used as a coordinate

    payload: a legal payload
        A legal "payload" value.


    Notes
    -----

    Construction of a element of this class relies on the `payload`
    argument already being a legal payload. Frequently, that will be
    will a instance of a `Payload` (see `fibertree.core.payload`). But
    because it is already a legal payload `__init__()` does not try to
    invoke `Payload.maybe_box()`.

    Iteration through an instance of this class results in the
    "coordinate" followed by the "payload".

    """
    def __init__(self, coord, payload):
        """__init__"""

        #
        # Set up logging
        #
        # self.logger = logging.getLogger('fibertree.core.coord_payload')


        self.coord = coord
        self.payload = Payload.maybe_box(payload)


    def __iter__(self):
        """__iter__"""

        yield self.coord
        yield self.payload

    #
    # Position based methods
    #
    def __getitem__(self, keys):
        """Index into the payload

        Do a `__getitem__()` on the payload of `self`.  Generally this
        will only be meaningful if the payload is a fiber. So see
        `Fiber.__getitem__()` for more information.

        Parameters
        ----------
        keys: single integer/slice or tuple of integers/slices
            The positions or slices in an n-D fiber

        Returns
        -------
        tuple or Fiber
            A tuple of a coordinate and payload or a Fiber of the slice

        Raises
        ------

        IndexError
            Index out of range

        TypeError
            Invalid key type

        """
        return self.payload.__getitem__(keys)


    def __setitem__(self, key, newvalue):
        """Index into a payload and update the value

        Do a `__setitem__()` on the payload of `self`.  Generally this
        will only be meaningful if the payload is a fiber. So see
        `Fiber.__setitem__()` for more information.

        Parameters
        ----------
        key: single integer
            The position in the fiber to be set

        newvalue: a CoordPayload or a payload value
            The coordinate/payload or just payload to assign

        Returns
        -------
        Nothing

        Raises
        ------

        IndexError
            Index out of range

        TypeError
            Invalid key type

        CoordinateError
            Invalid coordinate

        """

        self.payload.__setitem__(key, newvalue)

    #
    # Assignment operator
    #
    def __ilshift__(self, other):
        """Assign a new **boxed** value.

        Assigns a new value to a `Payload`. Since the normal
        Python assignment operator (=) will replace a pointer to a class
        rather than update a value in the class (i.e., the **boxed**
        value) we overload the "<<=" operator to assign a new **boxed**
        value to a `Payload`

        Parameters
        ----------
        other: Payload or scalar
            A value to assign as the new **boxed** value.

        Returns
        -------
        Nothing


        Examples
        --------

        ```
        >>> a = CoordPayload(1, 4)
        >>> print(a)
        CoordPayload(1, 4)
        >>> b = a
        >>> b <<= 6
        >>> print(a)
        CoordPayload(1, 6)
        >>> b = 8
        >>> print(a)
        CoordPayload(1, 6)
        ```

        Notes
        -----

        There is an analogous assignment operator for the `Payload`
        and `Fiber` classes, so one can "assign" a new value to a
        "payload" irrespective of whether the "payload" is a
        `Payload`, a `CoordPayload` or a `Fiber`.

        """
        if isinstance(other, CoordPayload):
            self.payload <<= other.payload
        else:
            self.payload <<= self.payload + other


    #
    # Arithmetic operations
    #
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

    #
    # Copy operation
    #
    def __deepcopy__(self, memo):
        """__deepcopy__

        Note: to ensure maintainability, we want to automatically copy
        everything. We use pickling because it is much more performant
        than the default deepcopy
        """
        return pickle.loads(pickle.dumps(self))

    #
    # Printing
    #
    def __repr__(self):
        """__repr__"""

        return str(f"CoordPayload(coord={self.coord}, payload={self.payload})")

#
# Pdoc stuff
#
__pdoc__ = {'CoordPayload.__getitem__':   True,
            'CoordPayload.__setitem__':   True,
            'CoordPayload.__ilshift__':   True,
           }

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
