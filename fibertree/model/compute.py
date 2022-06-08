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
    def numIsectLeaderFollower(dump, rank, leader):
        """
        Compute the number of intersection attempts with leader-follower
        intersection

        Parameters
        ----------

        dump: dict
            The statistics counted by `Metrics.dump()`

        rank: str
            The rank whose intersection tests we care about

        leader: int
            Tensor number of the leader

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        line = "Rank " + rank
        if leader % 2 == 0:
            first = str(leader)
            second = str(leader + 1)
        else:
            first = str(leader - 1)
            second = str(leader)

        succ = dump[line]["successful_intersect_" + first + "_" + second]
        unsucc = dump[line]["unsuccessful_intersect_tensor" + str(leader)]
        return succ + unsucc

    @staticmethod
    def numIsectSkipAhead(dump, rank, left=0):
        """ Compute the number of intersection attempts with skip-ahead
        intersection

        Parameters
        ----------

        dump: dict
            The statistics counted by `Metrics.dump()`

        rank: str
            The rank whose intersection tests we care about

        left: int
            Tensor number of the left tensor (default=0)

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        line = "Rank " + rank
        right = str(left + 1)
        left = str(left)
        both = left + "_" + right

        succ = dump[line]["successful_intersect_" + both]
        unsucc_a = dump[line]["unsuccessful_intersect_tensor" + left]
        unsucc_b = dump[line]["unsuccessful_intersect_tensor" + right]
        skipped = dump[line]["skipped_intersect_" + both]

        return succ + unsucc_a + unsucc_b - skipped


    @staticmethod
    def swapCount(tensor, depth, radix, next_latency):
        """Compute the number of swaps required at the given depth"""
        return Compute._swapCountTree(tensor.getRoot(), depth, radix, next_latency)

    @staticmethod
    def _swapCountTree(fiber, depth, radix, next_latency):
        """Compute the number of swaps required at the given depth"""
        swaps = 0

        # Recurse if necessary
        if depth > 0:
            depth -= 1
            for _, payload in fiber:
                swaps += Compute._swapCountTree(payload, depth, radix, next_latency)
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
