#cython: language_level=3
"""Payload

A class implementing a **boxed** value to use as a payload
of an element of a fiber.

"""
import logging
import pickle

from .metrics import Metrics

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.core.payload')


class Payload:
    """A class to hold the payload of an element of a fiber.

    This class supports to ability for a program operating on the
    payload of an element of a fiber to have a reference to that
    payload separate from the element containing it. Since, one needs
    to be able update that payload and see the update reflected in the
    fiber this class has the effect of **boxing** the payload for
    immutable types.

    Frequently, a `Payload` will appear as the payload of an element
    of a fiber as part of an instance of a `CoordPayload` (see
    `fibertree.core.coord_payload`).

    A substantial set of infix operators are provided that operate on
    a `Payload` and another `Payload` or a scalar value. These
    include:

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
    - Great than: (>)
    - Less than or equal: (<=)
    - Greater than or equal: (>=)

    In addition to the above operators one needs to be able to
    conveniently assign a new value to a `Payload`. Since the normal
    Python assignment operator (=) will replace a pointer to a class
    rather than update a value in the class (i.e., the **boxed**
    value) we overload the operator "<<=" to assign a new **boxed**
    value to a `Payload` (see `Payload.__ilshift__()`)

    Constructor
    -----------

    The `Payload` constructor creates an payload with an optionally
    given `value`.

    Parameters
    ----------
    value: immutable value, default=None
         The value to **box**.


    Notes
    -----

    Currently, a variety of immutable types, e.g., int, str and
    tuples, are **boxed** by the `Payload` class, while the `Fiber`
    class, which is the other common payload, is not **boxed** by this
    class. The `Payload.maybe_box()` method can be used to selectively
    wrap a fiber element's payload in the `Payload` class.

    Furthermore, to make it more convenient for a program to operate
    on an arbitrary type of fiber element payload, this class provides
    a variety of **static** methods that selectively peek inside the
    **box** to do their job. These include:

    - `Payload.isEmpty()`
    - `Payload.is_payload()`
    - `Payload.contains()`
    - `Payload.get()`

    """

    def __new__(cls, value=None):

        #
        # Since we do not wrap Fibers in a Payload, we check if we
        # just want to just return the Fiber.
        #
        if type(value).__name__ == "Fiber":
            return value

        #
        # Just handle regular Payload creation
        #
        self = super(Payload, cls).__new__(cls)
        self.__init__(value=value)
        return self


    def __init__(self, value=None):
        """__init__"""

        #
        # Set up logging
        #
        # self.logger = logging.getLogger('fibertree.core.payload')

        self.value = value

    def v(self):
        """Return the **boxed** value

        Parameters
        ----------
        None

        Returns
        -------
        value: some immutable type
            The **boxed** value

        """

        return self.value

    def __setattr__(self, name, value):
        """Set an attribute of the Payload.

        Allow users to set the **boxed** value as an attribute (.v) of
        an instance of the class.

        Examples
        --------

        ```
        >>> payload = Payload()
        >>> payload.v = 5
        ```

        Notes
        -----

        The `Payload.__ilshift__()` payload assignment operator (<<=)
        should be used in preference to this method.

        """

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
        """Check if a fiber element's payload is empty.

        Selectively look into the given argument (`p`) to see if it is
        empty. In essence, if the given argument is a `Payload` check
        if its **boxed** value is the **empty** value. If the given
        argument is a `Fiber`, then check if the fiber is empty.

        Parameters
        ----------
        p: a payload
            The payload of an element of a fiber.


        Returns
        -------
        is_empty: Boolean
            Whether `p` was **empty**.


        Todo
        ----

        This needs to work when **empty** is something other than zero.

        """

        if type(p).__name__ == "Fiber":
            return p.isEmpty()

        if isinstance(p, tuple):
            assert isinstance(p, tuple)

        if (p == 0):
            return True

        return False

#
#
# Transition methods
#
# Note: The following methods are used as part of a future transition
#       from Fibers holding a raw Fiber as a payload to that Fiber
#       being embedded in in a Payload object.
#
    @staticmethod
    def maybe_box(value):
        """Selectively **box** a value.

        Selectively wrap certain values in a `Payload` wrapper.
        Currently only certain immutable types are **boxed**, and
        notably `Fibers` are not boxed nor are values that are already
        a `Payload`.

        Parameters
        ----------
        value: any type
            A value to possibly be **boxed** as a `Payload`


        Returns
        -------
        maybe_boxed: updatable type
            A reference to a value that can be updated

        Todo
        ----

        For consistency maybe this should be maybeBox().

        """

        if isinstance(value, (bool, float, int, str, tuple, frozenset)):
            return Payload(value)

        return value


    @staticmethod
    def is_payload(payload):
        """Check if argument is a payload.

        Check if the given argument (`payload`) is potentially the
        payload of an element of a fiber.  In essence, check if the
        given argument is a `Payload` or a `Fiber`.

        Parameters
        ----------
        payload: a payload
            The potential payload of an element of a fiber.


        Returns
        -------
        is_payload: Boolean
            Whether `payload` was a `Payload` or `Fiber`

        Todo
        ----

        For consistency maybe this should be isPayload().

        """

        from .fiber import Fiber

        return isinstance(payload, (Payload, Fiber))


    @staticmethod
    def contains(payload, type):
        """Return whether `payload` is of type `type`.

        Selectively look into the given argument (`payload`) to see if
        it is of the requested type. In essence, if the given argument
        is a `Payload` check if its **boxed** value is of type
        `type`. If the given argument is a `Fiber`, the result is True
        if the caller is checking for a `Fiber`.

        Parameters
        ----------
        payload: a payload
            The payload of an element of a fiber.

        Returns
        -------
        contains: Boolean
            Whether `payload` was of type `type`.

        """

        assert type != Payload, "Cannot check for Payload type"

        if not isinstance(payload, Payload):
            return isinstance(payload, type)

        return isinstance(payload.value, type)


    @staticmethod
    def get(payload):
        """Return value of `payload`.

        Selectively look into the given argument (`payload`) and
        return its value.  In essence, if the given argument is a
        `Payload` return its **boxed** value.  If the given argument
        is a `Fiber`, then return it.

        Parameters
        ----------
        payload: a payload
            The payload of an element of a fiber.

        Returns
        -------
        value: any type
            The **boxed** value of a `Payload` or a `Fiber`.

        """

        if not isinstance(payload, Payload):
            return payload

        return payload.value

#
# Srtring operations
#

    def print(self, title=None):
        """print"""

        return self.value.print(title)


    def __format__(self, spec=""):
        """__format__"""

        if len(spec) > 0:
            return f"<{self.value:{spec}}>"
        else:
            return f"<{self.value}>"


    def __str__(self):
        """__str__"""

        return f"<{self.value.__str__()}>"


    def __repr__(self):
        """__repr__"""

        return f"Payload({self.value.__repr__()})"
#
# Arithmetic operations
#
    def __add__(self, other):
        """__add__"""

        if isinstance(other, Payload):
            ans = self.value + other.value
        else:
            ans = self.value + other

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_add", 1)

        return Payload(ans)

    def __radd__(self, other):
        """__radd__"""

        assert not isinstance(other, Payload)

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_add", 1)

        return Payload(other + self.value)

    def __iadd__(self, other):
        """__iadd__"""

        if isinstance(other, Payload):
            self.value = self.value + other.value
        else:
            self.value = self.value + other

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_add", 1)
            Metrics.incCount("Compute", "payload_update", 1)

        return self

    # Note: we use <<= in place of base '=' so this is a pure overwrite
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
        >>> a = Payload(4)
        >>> print(a)
        4
        >>> b = a
        >>> b <<= 6
        >>> print(a)
        6
        >>> b = 8
        >>> print(a)
        6
        ```

        Notes
        -----

        There is an analogous assignment operator for the `Fiber` and
        `CoordPayload` classes, so one can "assign" a new value to a
        "payload" irrespective of whether the "payload" is a
        `Payload`, `CoordPayload` or a `Fiber`.

        """

        if isinstance(other, Payload):
            self.value = other.value
        else:
            self.value = other
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

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_mul", 1)

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

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_mul", 1)

        return Payload(other * self.value)


    def __imul__(self, other):
        """__imul__"""

        if isinstance(other, Payload):
            self.value = self.value * other.value
        else:
            self.value = self.value * other

        # Collect metrics
        if Metrics.isCollecting():
            Metrics.incCount("Compute", "payload_mul", 1)
            Metrics.incCount("Compute", "payload_update", 1)

        return self


#
# Comparison operations
#
    def __eq__(self, other):
        """__eq__"""

        if isinstance(other, Payload):
            return self.value == other.value

        return self.value == other

    def __lt__(self, other):
        """__lt__"""

        if isinstance(other, Payload):
            return self.value < other.value

        return self.value < other

    def __le__(self, other):
        """__le__"""

        if isinstance(other, Payload):
            return self.value <= other.value

        return self.value <= other

    def __gt__(self, other):
        """__gt__"""

        if isinstance(other, Payload):
            return self.value > other.value

        return self.value > other

    def __ge__(self, other):
        """__ge__"""

        if isinstance(other, Payload):
            return self.value >= other.value

        return self.value >= other

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
# Conversion methods - to/from dictionaries
#

    @staticmethod
    def payload2dict(payload):
        """Return payload converted to dictionry or simple value"""

        from .fiber import Fiber

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


#
# Pdoc stuff
#
__pdoc__ = { 'Payload.payload2dict':   False,
             'Payload.__setattr__':    True,
             'Payload.__ilshift__':    True,
           }


if __name__ == "__main__":

    a = Payload(1)
    print("A = %s" % a)
    print("---")
