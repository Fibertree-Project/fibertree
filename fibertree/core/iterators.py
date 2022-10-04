#cython: language_level=3
"""Iterator

A module storing the implementations of all of the iterators of the Fiber class

"""

import bisect

from .coord_payload import CoordPayload
from .metrics import Metrics
from .payload import Payload

def __iter__(self, tick=True, start_pos=None):
    """__iter__"""
    if self.getOwner() is not None:
        fmt = self.getOwner().getFormat()
    elif self.getRankAttrs() is not None:
        fmt = self.getRankAttrs().getFormat()
    else:
        fmt = "C"

    if fmt == "C":
        return self.iterOccupancy(tick, start_pos=start_pos)
    elif fmt == "U":
        return self.iterShape(tick)
    else:
        raise ValueError("Unknown format")


def __reversed__(self):
    """Return reversed fiber"""

    assert not self.isLazy()

    for coord, payload in zip(reversed(self.coords),
                              reversed(self.payloads)):
        yield CoordPayload(coord, payload)

def iterOccupancy(self, tick=True, start_pos=None):
    """Iterate over non-default elements of the fiber

    Iterate over every non-default payload in the shape, returning a
    CoordPayload for each one

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter

    start_pos: Optional[int]
        Saved position to start iteration

    """
    return self.iterRange(None, None, tick=tick, start_pos=start_pos)

def iterShape(self, tick=True):
    """Iterate over fiber shape

    Iterate over every coordinate in the shape, returning a
    CoordPayload for each one, with a **default** value for
    empty payloads.

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter

    """
    return self.iterRangeShape(0, self.getShape(all_ranks=False), tick=tick)

def iterShapeRef(self, tick=True):
    """Iterate over fiber shape

    Iterate over every coordinate in the shape, returning a
    CoordPayload for each one, and creating elements for empty
    payloads.

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter

    """
    return self.iterRangeShapeRef(0, self.getShape(all_ranks=False), tick=tick)

def iterActive(self, tick=True, start_pos=None):
    """Iterate over the non-default elements within the fiber's active range

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter
    """
    return self.iterRange(*self.getActive(), tick=tick, start_pos=start_pos)

def iterActiveShape(self, tick=True):
    """Iterate over the fiber's active range, including default elements

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter
    """
    return self.iterRangeShape(*self.getActive(), tick=tick)

def iterActiveShapeRef(self, tick=True):
    """Iterate over the fiber's active range, creating default elements if they
    do not exist

    Parameters
    ----------
    tick: bool
        True if this iterator should tick the metrics counter
    """
    return self.iterRangeShapeRef(*self.getActive(), tick=tick)

def iterRange(self, start, end, tick=True, start_pos=None):
    """
    Iterate over the non-default elements within the given range

    Parameters
    ----------
    start: Optional[int]
        Beginning of range (inclusive); None implies no bound

    end: Optional[int]
        End of range (exclusive); None implies no bound

    tick: bool
        True if this iterator should tick the metrics counter

    start_pos: Optional[int]
        Saved position to start iteration

    """
    # Cannot save a position of a lazy fiber
    assert not self.isLazy() or start_pos is None

    # Get the iterator
    if self.isLazy():
        iter_ = self.iter()
    else:
        # Set i: the starting position
        start_pos = Payload.get(start_pos)
        if start_pos is not None:
            assert start_pos < len(self.coords)
            i = start_pos
        else:
            i = 0

        iter_ = ((self.coords[j], self.payloads[j])
                  for j in range(i, len(self.coords)))

    is_collecting, rank = _prep_metrics_inc(self)

    if is_collecting and tick:
        Metrics.registerRank(rank)

    for j, (coord, payload) in enumerate(iter_):
        # If we are outside the range, stop
        if end is not None and coord >= end:
            break

        # If we are within the range, emit the non-default elements
        elif start is None or coord >= start:
            if not Payload.isEmpty(payload):
                if start_pos is not None:
                    self.setSavedPos(i + j, distance=j)

                if is_collecting and tick:
                    Metrics.addUse(rank, coord)

                yield CoordPayload(coord, payload)

                if is_collecting and tick:
                    Metrics.incIter(rank)

        # Otherwise continue iterating untile we find the beginning of the
        # range

    if is_collecting and tick:
        Metrics.endIter(rank)

def iterRangeShape(self, start, end, step=1, tick=True):
    """Iterate over the given range, including default elements

    Parameters
    ----------
    start: int
        Beginning of range (inclusive)

    end: int
        End of range (exclusive)

    step: int
        Step of each iteration

    tick: bool
        True if this iterator should tick the metrics counter
    """
    assert not self.isLazy()

    is_collecting, rank = _prep_metrics_inc(self)

    if is_collecting and tick:
        self.registerRank(rank)

    for c in range(start, end, step):
        p = self.getPayload(c)
        yield CoordPayload(c, p)

        if is_collecting and tick:
            Metrics.incIter(rank)

    if is_collecting and tick:
        Metrics.endIter(rank)

def iterRangeShapeRef(self, start, end, step=1, tick=True):
    """Iterate over the given range, including default elements

    Parameters
    ----------
    start: int
        Beginning of range (inclusive)

    end: int
        End of range (exclusive)

    step: int
        Step of each iteration

    tick: bool
        True if this iterator should tick the metrics counter
    """
    assert not self.isLazy()

    is_collecting, rank = _prep_metrics_inc(self)

    if is_collecting and tick:
        self.registerRank(rank)

    for c in range(start, end, step):
        p = self.getPayloadRef(c)
        yield CoordPayload(c, p)

        if is_collecting and tick:
            Metrics.incIter(rank)

    if is_collecting and tick:
        Metrics.endIter(rank)


def _prep_metrics_inc(fiber):
    """Prepare to do a metrics increment

    Returns
    -------

    is_collecting: bool
        True if Metrics collection is on

    rank: str
        The name of the rank number to increment over
    """
    is_collecting = Metrics.isCollecting()
    rank = str(fiber.getRankAttrs().getId())

    return is_collecting, rank

#
# Dense coiterators
#
def coiterShape(fibers):
    """Co-iterate in a dense manner over the given fibers

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0

    return type(fibers[0]).coiterRangeShape(fibers, 0, fibers[0].getShape(all_ranks=False))

def coiterShapeRef(fibers):
    """Co-iterate in a dense manner over the given fibers, inserting any
    implicit payloads

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0

    return type(fibers[0]).coiterRangeShapeRef(fibers, 0, fibers[0].getShape(all_ranks=False))

def coiterActiveShape(fibers):
    """Co-iterate in a dense manner over the given fibers using the active
    range of the first fiber

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0

    return type(fibers[0]).coiterRangeShape(fibers, *fibers[0].getActive())

def coiterActiveShapeRef(fibers):
    """Co-iterate in a dense manner over the given fibers using the active
    range of the first fiber, inserting any implicit payloads

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0

    return type(fibers[0]).coiterRangeShapeRef(fibers, *fibers[0].getActive())

def coiterRangeShape(fibers, start, end, step=1):
    """Co-iterate in a dense manner over the given fibers using the given
    range

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    start: int
        Beginning of range (inclusive)

    end: int
        End of range (exclusive)

    step: int
        Coordinates per iteration

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0
    assert all(not fiber.isLazy() for fiber in fibers)

    class coiter_range_shape_iterator:
        fibers_ = fibers
        start_ = start
        end_ = end
        step_ = step

        def __iter__(self):
            for c in range(self.start_, self.end_, self.step_):
                payloads = tuple(fiber.getPayload(c) for fiber in self.fibers_)
                yield CoordPayload(c, payloads)

    fiber = fibers[0].fromIterator(coiter_range_shape_iterator)
    fiber.getRankAttrs().setId(fibers[0].getRankAttrs().getId())
    fiber.setActive((start, end))
    return fiber

def coiterRangeShapeRef(fibers, start, end, step=1):
    """Co-iterate in a dense manner over the given fibers using the given
    range, inserting any implicit payloads

    Parameters
    ----------

    fibers: List[Fiber]
        A list of fibers to coiterate over

    start: int
        Beginning of range (inclusive)

    end: int
        End of range (exclusive)

    step: int
        Coordinates per iteration

    Returns
    -------

    result: Fiber
        A fiber whose payloads are the payloads of the corresponding tuples

    """
    assert len(fibers) > 0
    assert all(not fiber.isLazy() for fiber in fibers)

    class coiter_range_shape_ref_iterator:
        fibers_ = fibers
        start_ = start
        end_ = end
        step_ = step

        def __iter__(self):
            for c in range(self.start_, self.end_, self.step_):
                payloads = tuple(fiber.getPayloadRef(c) for fiber in self.fibers_)
                yield CoordPayload(c, payloads)

    fiber = fibers[0].fromIterator(coiter_range_shape_ref_iterator)
    fiber.getRankAttrs().setId(fibers[0].getRankAttrs().getId())
    fiber.setActive((start, end))
    return fiber

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
    class intersection_iterator:
        nested = nested_result

        def __iter__(self):
            for c, np in self.nested.__iter__(tick=False):
                p = []
                while isinstance(Payload.get(np), tuple):
                    val = Payload.get(np)
                    p.append(val[1])
                    np = val[0]
                p.append(np)
                yield CoordPayload(c, tuple(reversed(p)))

    # Call the constructor via the first argument
    fiber = args[0].fromIterator(intersection_iterator)
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
    class union_iterator:
        nested = nested_result
        num_args = len(args)

        def __iter__(self):
            for c, np in self.nested:
                p = [None] * (self.num_args + 1)

                # This is the mask
                p[0] = ""
                for i in range(self.num_args - 1, 0, -1):
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

    fiber = args[0].fromIterator(union_iterator)
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

    class and_iterator:
        a_fiber = self
        b_fiber = other

        def __iter__(self):
            """
            Iterator simulating the intersection operator
            """
            is_collecting = Metrics.isCollecting()
            if is_collecting:
                rank = self.a_fiber.getRankAttrs().getId()
                line = "Rank " + rank
                a_label = str(Metrics.getLabel(rank))
                b_label = str(Metrics.getLabel(rank))
                both = a_label + "_" + b_label

                coord_a = "coordinate_read_tensor" + a_label
                coord_b = "coordinate_read_tensor" + b_label
                payload_a = "payload_read_tensor" + a_label
                payload_b = "payload_read_tensor" + b_label
                unsucc_a = "unsuccessful_intersect_tensor" + a_label
                unsucc_b = "unsuccessful_intersect_tensor" + b_label
                succ = "successful_intersect_" + both
                skipped = "skipped_intersect_" + both

                Metrics.incCount(line, coord_a, 1)
                Metrics.incCount(line, coord_b, 1)
                Metrics.incCount(line, payload_a, 0)
                Metrics.incCount(line, payload_b, 0)
                Metrics.incCount(line, unsucc_a, 0)
                Metrics.incCount(line, unsucc_b, 0)
                Metrics.incCount(line, succ, 0)
                Metrics.incCount(line, skipped, 0)

                skip = None

            # Get the iterators
            a = self.a_fiber.__iter__(tick=False)
            b = self.b_fiber.__iter__(tick=False)

            a_coord, a_payload = _get_next(a)
            b_coord, b_payload = _get_next(b)

            while not (a_coord is None or b_coord is None):
                if a_coord == b_coord:

                    if is_collecting:
                        # Increment the count metrics
                        Metrics.incCount(line, succ, 1)
                        Metrics.incCount(line, payload_a, 1)
                        Metrics.incCount(line, payload_b, 1)

                    yield a_coord, (a_payload, b_payload)

                    a_coord, a_payload = _get_next(a)
                    b_coord, b_payload = _get_next(b)

                    if is_collecting:
                        if a_coord is not None:
                            Metrics.incCount(line, coord_a, 1)

                        if b_coord is not None:
                            Metrics.incCount(line, coord_b, 1)

                    continue

                if a_coord < b_coord:
                    a_coord, a_payload = _get_next(a)

                    if is_collecting:
                        Metrics.incCount(line, unsucc_a, 1)

                        if skip == "A":
                            Metrics.incCount(line, skipped, 1)

                        if a_coord is not None:
                            Metrics.incCount(line, coord_a, 1)

                            if a_coord < b_coord:
                                skip = "A"
                            else:
                                skip = None
                    continue

                if a_coord > b_coord:
                    b_coord, b_payload = _get_next(b)

                    if is_collecting:
                        Metrics.incCount(line, unsucc_b, 1)

                        if skip == "B":
                            Metrics.incCount(line, skipped, 1)

                        if b_coord is not None:
                            Metrics.incCount(line, coord_b, 1)

                            if b_coord < a_coord:
                                skip = "B"
                            else:
                                skip = None

                    continue

            if is_collecting:
                if a_coord is None and b_coord is None:
                    Metrics.incCount(line, "same_last_coord_" + both, 1)
                else:
                    Metrics.incCount(line, "diff_last_coord_" + both, 1)

            return

    fiber = self.fromIterator(and_iterator)
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

    class or_iterator:
        a_fiber = self
        b_fiber = other

        def __iter__(self):
            a = self.a_fiber.__iter__()
            b = self.b_fiber.__iter__()

            a_coord, a_payload = _get_next(a)
            b_coord, b_payload = _get_next(b)

            while a_coord is not None and b_coord is not None:
                if a_coord == b_coord:
                    yield a_coord, ("AB", a_payload, b_payload)

                    a_coord, a_payload = _get_next(a)
                    b_coord, b_payload = _get_next(b)

                elif a_coord < b_coord:
                    b_default = self.b_fiber._createDefault()
                    yield a_coord, ("A", a_payload, b_default)
                    a_coord, a_payload = _get_next(a)

                # a_coord > b_coord
                else:
                    a_default = self.a_fiber._createDefault()
                    yield b_coord, ("B", a_default, b_payload)
                    b_coord, b_payload = _get_next(b)

            while a_coord is not None:
                b_default = self.b_fiber._createDefault()
                yield a_coord, ("A", a_payload, b_default)
                a_coord, a_payload = _get_next(a)

            while b_coord is not None:
                    a_default = self.a_fiber._createDefault()
                    yield b_coord, ("B", a_default, b_payload)
                    b_coord, b_payload = _get_next(b)

    result = self.fromIterator(or_iterator)
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

    class xor_iterator:
        a_fiber = self
        b_fiber = other

        def __iter__(self):
            a = self.a_fiber.__iter__()
            b = self.b_fiber.__iter__()

            a_coord, a_payload = _get_next(a)
            b_coord, b_payload = _get_next(b)

            while a_coord is not None and b_coord is not None:
                if a_coord == b_coord:
                    a_coord, a_payload = _get_next(a)
                    b_coord, b_payload = _get_next(b)

                elif a_coord < b_coord:
                    b_default = self.b_fiber._createDefault()
                    yield a_coord, ("A", a_payload, b_default)
                    a_coord, a_payload = _get_next(a)

                # a_coord > b_coord
                else:
                    a_default = self.a_fiber._createDefault()
                    yield b_coord, ("B", a_default, b_payload)
                    b_coord, b_payload = _get_next(b)

            while a_coord is not None:
                b_default = self.b_fiber._createDefault()
                yield a_coord, ("A", a_payload, b_default)
                a_coord, a_payload = _get_next(a)

            while b_coord is not None:
                a_default = self.a_fiber._createDefault()
                yield b_coord, ("B", a_default, b_payload)
                b_coord, b_payload = _get_next(b)

    result = self.fromIterator(xor_iterator)
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

    class lshift_iterator:
        a_fiber = self
        b_fiber = other

        def __iter__(self):
            """
            Iterator simulating the populate operator
            """

            is_collecting = Metrics.isCollecting()

            if is_collecting:
                rank = self.a_fiber.getRankAttrs().getId()
                line = "Rank " + rank
                a_label = str(Metrics.getLabel(rank))
                b_label = str(Metrics.getLabel(rank))

                coord_a = "coordinate_read_tensor" + a_label
                coord_b = "coordinate_read_tensor" + b_label
                payload_a = "payload_read_tensor" + a_label
                payload_b = "payload_read_tensor" + b_label
                append = "coord_payload_append_tensor" + a_label
                insert = "coord_payload_insert_tensor" + a_label

                Metrics.incCount(line, coord_a, 0)
                Metrics.incCount(line, coord_b, 0)
                Metrics.incCount(line, payload_a, 0)
                Metrics.incCount(line, payload_b, 0)
                Metrics.incCount(line, append, 0)
                Metrics.incCount(line, insert, 0)

            # Add coordinates/payloads to a_fiber where necessary
            maybe_remove = False
            b = self.b_fiber.__iter__(tick=False)
            for b_coord, b_payload in b:
                a_payload = self.a_fiber.getPayload(b_coord, allocate=False)

                if is_collecting:
                    Metrics.incCount(line, coord_b, 1)
                    Metrics.incCount(line, payload_b, 1)

                if a_payload is None:
                    if is_collecting:
                        if self.a_fiber.maxCoord() is None \
                                or self.a_fiber.maxCoord() < b_coord:
                            Metrics.incCount(line, append, 1)
                        else:
                            Metrics.incCount(line, insert, 1)

                    # Do not actually insert the payload into the tensor
                    a_payload = self.a_fiber._create_payload(b_coord)
                    maybe_remove = True

                elif is_collecting:
                    Metrics.incCount(line, coord_a, 1)
                    Metrics.incCount(line, payload_a, 1)
                    maybe_remove = False

                else:
                    maybe_remove = False

                yield b_coord, (a_payload, b_payload)

                # If this was never updated, remove it
                if maybe_remove and (isinstance(a_payload, type(self.a_fiber)) and \
                        len(a_payload) == 0) or \
                        (not isinstance(a_payload, type(self.a_fiber)) and \
                        a_payload == self.a_fiber.getDefault()):
                    # Clear the fiber
                    index = bisect.bisect_left(self.a_fiber.coords, b_coord)
                    del self.a_fiber.coords[index]
                    del self.a_fiber.payloads[index]

                    # Remove the payload from its owning rank (if relevant)
                    if self.a_fiber.getOwner() is not None and \
                            self.a_fiber.getOwner().getNextRank() is not None:
                        popped = self.a_fiber.getOwner().getNextRank().pop()

                        # Make sure that the rank was not modified in the
                        # middle and we actually popped off the correct fiber
                        assert id(a_payload) == id(popped)


            return

    fiber = self.fromIterator(lshift_iterator)
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

    class sub_iterator:
        a_fiber = self
        b_fiber = other

        def __iter__(self):
            a = self.a_fiber.__iter__()
            b = self.b_fiber.__iter__()

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

    result = self.fromIterator(sub_iterator)
    result._setDefault(self.getDefault())
    result.getRankAttrs().setId(self.getRankAttrs().getId())

    return result


