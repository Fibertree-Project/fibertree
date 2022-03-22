#cython: language_level=3
"""Iterator

A module storing the implementations of all of the iterators of the Fiber class

"""

from .coord_payload import CoordPayload
from .metrics import Metrics
from .payload import Payload

def __iter__(self):
    """__iter__"""
    if self.getOwner() is not None:
        fmt = self.getOwner().getFormat()
    elif self.getRankAttrs() is not None:
        fmt = self.getRankAttrs().getFormat()
    else:
        fmt = "C"

    if fmt == "C":
        return self.iterOccupancy()
    elif fmt == "U":
        return self.iterShape()
    else:
        raise ValueError("Unknown format")


def __reversed__(self):
    """Return reversed fiber"""

    assert not self.isLazy()

    for coord, payload in zip(reversed(self.coords),
                              reversed(self.payloads)):
        yield CoordPayload(coord, payload)

def iterOccupancy(self, tick=False):
    """Iterate over non-default elements of the fiber

    Iterate over every non-default payload in the shape, returning a
    CoordPayload for each one

    Parameters
    ----------
    None
    """
    is_collecting, line = _prep_metrics_inc(self)

    if self.isLazy():
        for coord, payload in self.iter:
            self.coords.append(coord)
            self.payloads.append(payload)
            yield CoordPayload(coord, payload)

            if is_collecting and tick:
                Metrics.incIter(line)

    else:
        for coord, payload in zip(self.coords, self.payloads):
            if not Payload.isEmpty(payload):
                yield CoordPayload(coord, payload)

                if is_collecting and tick:
                    Metrics.incIter(line)

    if is_collecting and tick:
        Metrics.clrIter(line)

def iterShape(self, tick=False):
    """Iterate over fiber shape

    Iterate over every coordinate in the shape, returning a
    CoordPayload for each one, with a **default** value for
    empty payloads.

    Parameters
    ----------
    None

    """
    assert not self.isLazy()

    is_collecting, line = _prep_metrics_inc(self)

    for c in range(self.getShape(all_ranks=False)):
        p = self.getPayload(c)
        yield CoordPayload(c, p)

        if is_collecting and tick:
            Metrics.incIter(line)

    if is_collecting and tick:
        Metrics.clrIter(line)

def iterShapeRef(self, tick=False):
    """Iterate over fiber shape

    Iterate over every coordinate in the shape, returning a
    CoordPayload for each one, and creating elements for empty
    payloads.

    Parameters
    ----------
    None

    """

    assert not self.isLazy()

    is_collecting, line = _prep_metrics_inc(self)

    for c in range(self.getShape(all_ranks=False)):
        p = self.getPayloadRef(c)
        yield CoordPayload(c, p)

        if is_collecting and tick:
            Metrics.incIter(line)

    if is_collecting and tick:
        Metrics.clrIter(line)

def _prep_metrics_inc(fiber):
    """Prepare to do a metrics increment

    Returns
    -------

    is_collecting: bool
        True if Metrics collection is on

    line: str
        The name of the line number to increment over
    """
    is_collecting = Metrics.isCollecting()

    if fiber.getOwner() is None:
        line = "Rank Unknown"
    else:
        line = "Rank " + str(fiber.getOwner().getId())

    return is_collecting, line


#
# Aggretated intersection/union methods
#
def intersection(*args):
    """Intersect a set of fibers.

    Create a new fiber containing all the coordinates that are
    common to **all** the fibers in `args` and for each of those
    coordinates create a payload that is the combination of the
    payloads of all the input fibers. Note, however, unlike a
    sequence of two-operand intersections (see Fiber.__and__()`)
    the payloads are combined together in one long `tuple`.

    Parameters
    ----------

    args: list of Fibers
        The set of fibers to intersect

    Returns
    -------

    result: Fiber
        A fiber containing the intersection of all the input fibers.


    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """

    nested_result = args[0] & args[1]

    for arg in args[2:]:
        nested_result = nested_result & arg

    # Lazy implementation
    def iterator(nested):
        for c, np in nested:
            p = []
            while isinstance(np, tuple):
                p.append(np[1])
                np = np[0]
            p.append(np)
            yield CoordPayload(c, tuple(reversed(p)))

    # Call the constructor via the first argument
    fiber = args[0].fromIterator(iterator(nested_result))
    fiber.getRankAttrs().setId(args[0].getRankAttrs().getId())
    return fiber

def union(*args):
    """Union a set of fibers.

    Create a new fiber containing the coordinates that exist in
    **any** of the fibers in `args` and for each of those
    coordinates create a payload that is the combination of the
    payloads of all the input fibers. Note, however, unlike a
    sequence of two-operand unions (see `Fiber.__or__()`) the
    payloads are combined together in one long `tuple` with a mask
    at the begining indicating all the fibers that had a non-empty
    payload at that coordinate.

    Parameters
    ----------

    args: list of Fibers
        The set of fibers to union

    Returns
    -------

    result: Fiber
        A fiber containing the union of all the input fibers.

    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """

    nested_result = args[0] | args[1]

    for arg in args[2:]:
        nested_result = nested_result | arg

    # Lazy implementation
    def iterator(nested, num_args):
        for c, np in nested:
            p = [None] * (num_args + 1)

            # This is the mask
            p[0] = ""
            for i in range(num_args - 1, 0, -1):
                if isinstance(np, Payload):
                    np = np.v()

                ab_mask = np[0]
                if "B" in ab_mask:
                    p[0] = chr(ord("A") + i) + p[0]

                p[i + 1] = np[2]
                np = np[1]

            if "A" in ab_mask:
                p[0] = "A" + p[0]

            p[1] = np
            yield CoordPayload(c, tuple(p))

    fiber = args[0].fromIterator(iterator(nested_result, len(args)))
    fiber._setDefault(tuple([""]+[arg.getDefault() for arg in args]))
    fiber.getRankAttrs().setId(args[0].getRankAttrs().getId())
    return fiber
#
# Private functions for used in merge methods
#
def _get_next(iter_):
    """get_next"""

    try:
        coord, payload = next(iter_)
    except StopIteration:
        return (None, None)

    return CoordPayload(coord, payload)

#
# Merge methods
#
def __and__(self, other):
    """Two-operand intersection

    Return the intersection of `self` and `other` by considering
    all possible coordinates and returning a fiber consisting of
    payloads containing a tuple of the payloads of the inputs for
    coordinates where the following truth table returns True:

    ```
                     coordinate not     |      coordinate
                    present in `other`  |    present in `other`
                +-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    not present |         False         |        False          |
    in `self`   |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    present in  |         False         |        True           |
    `self`      |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
    ```

    Parameters
    ----------
    other: Fiber
        A fiber to intersect with the current fiber


    Returns
    --------
    result: generator
        A generator yielding the coordinate-payload pairs according to the
        intersection rules

    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """

    assert self._ordered and self._unique


    def iterator(a_fiber, b_fiber):
        """
        Iterator simulating the intersection operator
        """

        # Get the iterators
        a = a_fiber.__iter__()
        b = b_fiber.__iter__()
        next_b = lambda: _get_next(b)


        a_coord, a_payload = _get_next(a)
        b_coord, b_payload = _get_next(b)

        line = "Rank " + a_fiber.getRankAttrs().getId()
        is_collecting = Metrics.isCollecting()

        if is_collecting:
            Metrics.incCount(line, "coordinate_read_tensor0", 1)
            Metrics.incCount(line, "coordinate_read_tensor1", 1)
            Metrics.incCount(line, "unsuccessful_intersect_tensor0", 0)
            Metrics.incCount(line, "unsuccessful_intersect_tensor1", 0)
            Metrics.incCount(line, "skipped_intersect", 0)

            skip = None

        a_collecting = a_fiber.getRankAttrs().getCollecting()
        b_collecting = b_fiber.getRankAttrs().getCollecting()

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:

                if is_collecting:
                    # Increment the count metrics
                    Metrics.incCount(line, "successful_intersect", 1)
                    Metrics.incCount(line, "payload_read_tensor0", 1)
                    Metrics.incCount(line, "payload_read_tensor1", 1)

                    # If we are collecting metrics and this is our first time
                    # through the loop, check if it is the inner loop

                yield a_coord, (a_payload, b_payload)

                if is_collecting:
                    # Track all reuses of the element
                    start_iter = Metrics.getIter()
                    Metrics.incIter(line)

                    if a_collecting:
                        a_fiber._addUse(a_coord, start_iter)

                    if b_collecting:
                        b_fiber._addUse(b_coord, start_iter)


                a_coord, a_payload = _get_next(a)
                b_coord, b_payload = _get_next(b)

                if is_collecting:
                    if a_coord is not None:
                        Metrics.incCount(line, "coordinate_read_tensor0", 1)

                    if b_coord is not None:
                        Metrics.incCount(line, "coordinate_read_tensor1", 1)

                continue

            if a_coord < b_coord:
                a_coord, a_payload = _get_next(a)

                if is_collecting:
                    Metrics.incCount(line, "unsuccessful_intersect_tensor0", 1)

                    if skip == "A":
                        Metrics.incCount(line, "skipped_intersect", 1)

                    if a_coord is not None:
                        Metrics.incCount(line, "coordinate_read_tensor0", 1)

                        if a_coord < b_coord:
                            skip = "A"
                        else:
                            skip = None
                continue

            if a_coord > b_coord:
                b_coord, b_payload = _get_next(b)

                if is_collecting:
                    Metrics.incCount(line, "unsuccessful_intersect_tensor1", 1)

                    if skip == "B":
                        Metrics.incCount(line, "skipped_intersect", 1)

                    if b_coord is not None:
                        Metrics.incCount(line, "coordinate_read_tensor1", 1)

                        if b_coord < a_coord:
                            skip = "B"
                        else:
                            skip = None

                continue

        if is_collecting:
            Metrics.clrIter(line)

            if a_coord is None and b_coord is None:
                Metrics.incCount(line, "same_last_coord", 1)
            else:
                Metrics.incCount(line, "diff_last_coord", 1)

        return

    fiber = self.fromIterator(iterator(self, other))
    fiber.getRankAttrs().setId(self.getRankAttrs().getId())
    return fiber

def __or__(self, other):
    """__or__

    Return the union of `self` and `other` by considering all possible
    coordinates and returning a fiber consisting of payloads containing
    a tuple of the payloads of the inputs for coordinates where the
    following truth table returns True:


    ```
                     coordinate not     |      coordinate
                    present in `other`  |    present in `other`
                +-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    not present |         False         |        True           |
    in `self`   |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    present in  |         True          |        True           |
    `self`      |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
    ```

    Parameters
    ----------
    other: Fiber
        A fiber to union with the current fiber


    Returns
    --------
    result: Fiber
        A fiber created according to the union rules

    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """


    assert self._ordered and self._unique

    def iterator(a_fiber, b_fiber):
        a = a_fiber.__iter__()
        b = b_fiber.__iter__()

        a_coord, a_payload = _get_next(a)
        b_coord, b_payload = _get_next(b)

        while a_coord is not None and b_coord is not None:
            if a_coord == b_coord:
                yield a_coord, ("AB", a_payload, b_payload)

                a_coord, a_payload = _get_next(a)
                b_coord, b_payload = _get_next(b)

            elif a_coord < b_coord:
                b_default = b_fiber._createDefault()
                yield a_coord, ("A", a_payload, b_default)
                a_coord, a_payload = _get_next(a)

            # a_coord > b_coord
            else:
                a_default = a_fiber._createDefault()
                yield b_coord, ("B", a_default, b_payload)
                b_coord, b_payload = _get_next(b)

        while a_coord is not None:
            b_default = b_fiber._createDefault()
            yield a_coord, ("A", a_payload, b_default)
            a_coord, a_payload = _get_next(a)

        while b_coord is not None:
                a_default = a_fiber._createDefault()
                yield b_coord, ("B", a_default, b_payload)
                b_coord, b_payload = _get_next(b)

    result = self.fromIterator(iterator(self, other))
    result._setDefault(("", self.getDefault(), other.getDefault()))
    result.getRankAttrs().setId(self.getRankAttrs().getId())

    return result


def __xor__(self, other):
    """__xor__

    Return the xor of `self` and `other` by considering all possible
    coordinates and returning a fiber consisting of payloads containing
    a tuple of the payloads of the inputs for coordinates where the
    following truth table returns True:


    ```
                     coordinate not     |      coordinate
                    present in `other`  |    present in `other`
                +-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    not present |         False         |        True           |
    in `self`   |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    present in  |         True          |        False          |
    `self`      |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
    ```

    Parameters
    ----------
    other: Fiber
        A fiber to xor with the current fiber


    Returns
    --------
    result: Fiber
        A fiber created according to the xor rules

    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """

    assert self._ordered and self._unique

    def iterator(a_fiber, b_fiber):
        a = a_fiber.__iter__()
        b = b_fiber.__iter__()

        a_coord, a_payload = _get_next(a)
        b_coord, b_payload = _get_next(b)

        while a_coord is not None and b_coord is not None:
            if a_coord == b_coord:
                a_coord, a_payload = _get_next(a)
                b_coord, b_payload = _get_next(b)

            elif a_coord < b_coord:
                b_default = b_fiber._createDefault()
                yield a_coord, ("A", a_payload, b_default)
                a_coord, a_payload = _get_next(a)

            # a_coord > b_coord
            else:
                a_default = a_fiber._createDefault()
                yield b_coord, ("B", a_default, b_payload)
                b_coord, b_payload = _get_next(b)

        while a_coord is not None:
            b_default = b_fiber._createDefault()
            yield a_coord, ("A", a_payload, b_default)
            a_coord, a_payload = _get_next(a)

        while b_coord is not None:
            a_default = a_fiber._createDefault()
            yield b_coord, ("B", a_default, b_payload)
            b_coord, b_payload = _get_next(b)

    result = self.fromIterator(iterator(self, other))
    result._setDefault(("", self.getDefault(), other.getDefault()))
    result.getRankAttrs().setId(self.getRankAttrs().getId())

    return result



def __lshift__(self, other):
    """Fiber assignment

    Return the "assignment" of `other` to `self` by considering
    all possible coordinates and returning a fiber consisting of
    payloads containing a tuple of the payloads of the inputs for
    coordinates where the following truth table returns True:


    ```
                     coordinate not     |      coordinate
                    present in `other`  |    present in `other`
                +-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    not present |         False         |        True           |
    in `self`   |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    present in  |         False         |        True           |
    `self`      |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
    ```

    Parameters
    ----------
    other: Fiber
        A fiber to assign into the current fiber


    Returns
    --------
    result: generator
        A generator yielding the coordinate-payload pairs according to the
        assignment rules


    Notes
    ------

    An explicit zero in the input will NOT generate a corresponding
    coordinate in the output!

    """
    assert not self.isLazy()

    def iterator(a_fiber, b_fiber):
        """
        Iterator simulating the populate operator
        """

        line = "Rank " + a_fiber.getRankAttrs().getId()

        is_collecting = Metrics.isCollecting()

        # Add coordinates/payloads to a_fiber where necessary
        maybe_insert = False
        a_collecting = a_fiber.getRankAttrs().getCollecting()
        b_collecting = b_fiber.getRankAttrs().getCollecting()

        for b_coord, b_payload in b_fiber:
            a_payload = self.getPayload(b_coord, allocate=False)

            if is_collecting:
                Metrics.incCount(line, "coordinate_read_tensor1", 1)
                Metrics.incCount(line, "payload_read_tensor1", 1)

            if a_payload is None:
                if is_collecting:
                    if self.maxCoord() is None or self.maxCoord() < b_coord:
                        Metrics.incCount(line, "coord_payload_append_tensor0", 1)
                    else:
                        Metrics.incCount(line, "coord_payload_insert_tensor0", 1)

                # Do not actually insert the payload into the tensor
                a_payload = self._createDefault()
                maybe_insert = True
            elif is_collecting:
                Metrics.incCount(line, "coordinate_read_tensor0", 1)
                Metrics.incCount(line, "payload_read_tensor0", 1)
                maybe_insert = False

            else:
                maybe_insert = False


            yield b_coord, (a_payload, b_payload)

            # Only plan to insert the payload if it is non-zero
            if maybe_insert and ((isinstance(a_payload, type(self)) \
                        and len(a_payload) > 0) \
                    or (not isinstance(a_payload, type(self)) \
                        and a_payload != self.getDefault())):
                self._create_payload(b_coord, a_payload)

            if is_collecting:
                start_iter = Metrics.getIter()
                Metrics.incIter(line)

                if a_collecting:
                    a_fiber._addUse(b_coord, start_iter)

                if b_collecting:
                    b_fiber._addUse(b_coord, start_iter)

        if is_collecting:
            Metrics.clrIter(line)

        return

    fiber = self.fromIterator(iterator(self, other))
    fiber.getRankAttrs().setId(self.getRankAttrs().getId())
    return fiber

def __sub__(self, other):
    """__sub__

    Return the "difference" of `other` from `self` by considering
    all possible coordinates and returning a fiber consisting of
    payloads containing a tuple of the payloads of the inputs for
    coordinates where the following truth table returns True:


    ```
                     coordinate not     |      coordinate
                    present in `other`  |    present in `other`
                +-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    not present |         False         |        False          |
    in `self`   |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
                |                       |                       |
    coordinate  |                       |                       |
    present in  |          True         |        False          |
    `self`      |                       |                       |
                |                       |                       |
    ------------+-----------------------+-----------------------+
    ```

    Parameters
    ----------
    other: Fiber
        A fiber to subtract from the current fiber


    Returns
    --------
    result: Fiber
        A fiber created according to the subtraction rules


    Note
    ----

    Currently only supported for "ordered", "unique" fibers.

    """


    assert self._ordered and self._unique

    def iterator(a_fiber, b_fiber):
        a = a_fiber.__iter__()
        b = b_fiber.__iter__()

        a_coord, a_payload = _get_next(a)
        b_coord, b_payload = _get_next(b)

        while a_coord is not None and b_coord is not None:
            if a_coord == b_coord:
                a_coord, a_payload = _get_next(a)
                b_coord, b_payload = _get_next(b)

            elif a_coord < b_coord:
                yield a_coord, a_payload
                a_coord, a_payload = _get_next(a)

            # a_coord > b_coord:
            else:
                b_coord, b_payload = _get_next(b)

        while a_coord is not None:
            yield a_coord, a_payload
            a_coord, a_payload = _get_next(a)

    result = self.fromIterator(iterator(self, other))
    result._setDefault(self.getDefault())
    result.getRankAttrs().setId(self.getRankAttrs().getId())

    return result


