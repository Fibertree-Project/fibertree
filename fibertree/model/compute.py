#cython: language_level=3
"""
Compute the number operations executed
"""
import bisect

from fibertree import Tensor

class Compute:
    """YS
    """

    @staticmethod
    def opCount(dump, op):
        """Compute the number of operations executed by this kernel """
        metric = "payload_" + op
        if(metric in dump["Compute"].keys()):
            return dump["Compute"][metric]
        else:
            return 0

    @staticmethod
    def lfCount(dump, rank, leader):
        """
        Compute the number of intersection attempts with leader-follower
        intersection

        leader is 0 or 1 depending on which tensor the leader is.
        """

        line = "Rank " + rank
        l = "tensor" + str(leader)
        metric = "unsuccessful_intersect_" + l
        return dump[line]["successful_intersect"] + dump[line][metric]

    @staticmethod
    def skipCount(dump, rank):
        """
        Compute the number of intersection attempts with skip-ahead
        intersection
        """
        line = "Rank " + rank
        total = dump[line]["successful_intersect"] + dump[line]["unsuccessful_intersect_tensor0"] + dump[line]["unsuccessful_intersect_tensor1"]
        skipped = dump[line]["skipped_intersect"]
        return total - skipped

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
