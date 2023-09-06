#cython: language_level=3
"""
Compute the number operations executed
"""
import bisect

from fibertree import Tensor

class Compute:
    """Class for storing all compute counting methods

    Note: All methods in this class are static. It is meaningless to
    instantiate a Compute object
    """

    def __init__(self):
        """Do not call!"""
        raise NotImplementedError

    @staticmethod
    def numIters(trace):
        """
        Compute the number of iterations (lines) in this trace
        """
        with open(trace, "r") as f:
            f.readline()

            iters = 0
            while f.readline():
                iters += 1

        return iters

    @staticmethod
    def numOps(dump, op):
        """Compute the number of operations executed by this kernel """
        metric = "payload_" + op
        if(metric in dump["Compute"].keys()):
            return dump["Compute"][metric]
        else:
            return 0

    @staticmethod
    def numSwaps(tensor, depth, radix, next_latency):
        """Compute the number of swaps required at the given depth

        Parameters
        ----------

        tensor: Tensor
            The tensor being swapped

        depth: int
            The depth of the swap

        radix: Union[int, "N"]
            The radix of the merger

        next_latency: Union[int, "N"]
            The latency to get the next element

        Returns
        -------

        num_swaps: int
            The number of cycles required to perform the swap
        """
        return Compute._numSwapsTree(tensor.getRoot(), depth, radix, next_latency)

    @staticmethod
    def _numSwapsTree(fiber, depth, radix, next_latency):
        """Compute the number of swaps required at the given depth"""
        swaps = 0

        # Recurse if necessary
        if depth > 0:
            depth -= 1
            for _, payload in fiber:
                swaps += Compute._numSwapsTree(payload, depth, radix, next_latency)
            return swaps

        # Otherwise merge
        coords = []
        for _, payload in fiber:
            coords.append(sorted([-c for c in payload.getCoords()]))

        while len(coords) > 1:
            new = []
            if radix > len(coords):
                radix = len(coords)

            for i in range(0, len(coords), radix):
                end = min(i + radix, len(coords))
                ops, merged = Compute._merge(coords[i:end], radix, next_latency)

                swaps += ops
                new.append(merged)

            coords = new

        return swaps

    @staticmethod
    def _merge(coords, radix, next_latency):
        """
        Merge sorted lists of coordinates into a single list

        All coordinates are negated to work with list.pop() and bisect
        """
        # If we have a finite next latency, use that
        if isinstance(next_latency, int):
            merged = [c for list_ in coords for c in list_]
            merged.sort()
            return next_latency * (len(coords) + len(merged)), merged

        # Otherwise, merge incrementally
        # First get the heads
        head = []
        compares = 0

        # First insert all fibers
        for i, list_ in enumerate(coords):
            elem = (list_.pop(), i)
            j = bisect.bisect_right(head, elem)
            compares += len(head) - j + 1
            head.insert(j, elem)

        # Now build the result
        merged = []
        while head:
            elem = head.pop()
            merged.append(elem[0])
            if len(coords[elem[1]]) == 0:
                continue

            new = (coords[elem[1]].pop(), elem[1])
            j = bisect.bisect_right(head, new)
            compares += len(head) - j + 1

            head.insert(j, new)

        merged.sort()

        return compares, merged

