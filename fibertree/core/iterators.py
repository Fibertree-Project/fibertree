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

