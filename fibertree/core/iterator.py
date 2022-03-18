#cython: language_level=3
"""Iterator

A module storing the implementations of all of the iterators of the Fiber class

"""

from .metrics import Metrics
from .rank import Rank

class Iterator:
    @staticmethod
    def iterOccupancy(fiber, tick=False):
        """Iterate over non-default elements of the fiber

        Iterate over every non-default payload in the shape, returning a
        CoordPayload for each one

        Parameters
        ----------
        None
        """
        is_collecting, line = Iterator._prep_metrics_inc()

        if fiber.isLazy():
            for coord, payload in fiber.iter:
                fiber.coords.append(coord)
                fiber.payloads.append(payload)
                yield CoordPayload(coord, payload)

                if is_collecting and tick:
                    Metrics.incIter(line)

        else:
            for coord, payload in zip(fiber.coords, fiber.payloads):
                if not Payload.isEmpty(payload):
                    yield CoordPayload(coord, payload)

                    if is_collecting and tick:
                        Metrics.incIter(line)

        if is_collecting and tick:
            Metrics.clrIter(line)

    @staticmethod
    def iterShape(fiber, tick=False):
        """Iterate over fiber shape

        Iterate over every coordinate in the shape, returning a
        CoordPayload for each one, with a **default** value for
        empty payloads.

        Parameters
        ----------
        None

        """
        assert not fiber.isLazy()

        is_collecting, line = Iterator._prep_metrics_inc()

        for c in range(fiber.getShape(all_ranks=False)):
            p = fiber.getPayload(c)
            yield CoordPayload(c, p)

            if is_collecting and tick:
                Metrics.incIter(line)

        if is_collecting and tick:
            Metrics.clrIter(line)

    @staticmethod
    def iterShapeRef(fiber, tick=False):
        """Iterate over fiber shape

        Iterate over every coordinate in the shape, returning a
        CoordPayload for each one, and creating elements for empty
        payloads.

        Parameters
        ----------
        None

        """

        assert not fiber.isLazy()

        is_collecting, line = Iterator._prep_metrics_inc()

        for c in range(fiber.getShape(all_ranks=False)):
            p = fiber.getPayloadRef(c)
            yield CoordPayload(c, p)

            if is_collecting and tick:
                Metrics.incIter(line)

        if is_collecting and tick:
            Metrics.clrIter(line)

    @staticmethod
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
            line = "Rank " + fiber.getOwner().getId()

        return is_collecting, line

