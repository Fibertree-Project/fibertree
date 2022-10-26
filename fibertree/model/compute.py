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
    def numOps(dump, op):
        """Compute the number of operations executed by this kernel """
        metric = "payload_" + op
        if(metric in dump["Compute"].keys()):
            return dump["Compute"][metric]
        else:
            return 0

    @staticmethod
    def numIsectLeaderFollower(trace_fn, leader):
        """
        Compute the number of intersection attempts with leader-follower
        intersection

        Parameters
        ----------

        trace_fn: str
            The filename of the intersection

        leader: int
            Tensor number of the leader

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        with open(trace_fn, "r") as f:
            cols = f.readline()[:-1].split(",")
            ind = cols.index(str(leader) + "_match")

            isects = 0
            for line in f.readlines():
                data = line[:-1].split(",")
                if data[ind] == "True":
                    isects += 1

        return isects

    @staticmethod
    def numIsectSkipAhead(trace_fn):
        """ Compute the number of intersection attempts with skip-ahead
        intersection

        Parameters
        ----------

        trace_fn: str
            The filename of the intersection

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        with open(trace_fn, "r") as f:
            isects = 0
            curr = None
            for line in f.readlines():
                data = line[:-1].split(",")
                a_match, b_match = tuple(match == "True" for match in data[-2:])

                # If both matched, there is nothing to skip
                if a_match and b_match:
                    curr = None
                    isects += 1

                # If only A matched, intersect if not skipped
                elif a_match and curr != 0:
                    curr = 0
                    isects += 1

                # If only B matched, intersect if not skipped
                elif b_match and curr != 1:
                    curr = 1
                    isects += 1

        return isects


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
