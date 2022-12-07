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
    def numIsectLeaderFollower(leader_fn):
        """
        Compute the number of intersection attempts with leader-follower
        intersection

        Parameters
        ----------

        leader_fn: str
            The filename of the access trace of the leader

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        with open(leader_fn, "r") as f:
            # Throw away the header
            f.readline()

            isects = 0
            line = f.readline()
            while line:
                isects += 1
                line = f.readline()

        return isects

    @staticmethod
    def numIsectNaive(fn0, fn1):
        """ Compute the number of intersection attempts with a naive
        intersection unit

        Parameters
        ----------

        fn0, fn1: str
            The filenames of the intersection traces

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        def get_data(f, stamp_len):
            line = f.readline()
            if line:
                data = tuple(int(val) for val in line[:-1].split(",")[:-1])
            else:
                data = (float("inf"),)

            return line, data, data[stamp_len:-1]

        def get_next(f0, line0, data0, f1, line1, data1, stamp_len, advance0, advance1):
            if data0:
                stamp0 = data0[stamp_len:-1]
            else:
                stamp0 = None

            if data1:
                stamp1 = data1[stamp_len:-1]
            else:
                stamp1 = None

            if advance0:
                old_stamp0 = stamp0
                line0, data0, stamp0 = get_data(f0, stamp_len)

            if advance1:
                old_stamp1 = stamp1
                line1, data1, stamp1 = get_data(f1, stamp_len)

            while line0 and line1:
                if stamp0 == stamp1:
                    break

                elif stamp0 < stamp1:
                    line0, data0, stamp0 = get_data(f0, stamp_len)

                # stamp0 > stamp1
                else:
                    line1, data1, stamp1 = get_data(f1, stamp_len)

            return line0, data0, line1, data1


        with open(fn0, "r") as f0, open(fn1, "r") as f1:
            # Throw away headers
            line0 = f0.readline()
            f1.readline()

            isects = 0

            stamp_len = (len(line0.split(",")) - 1) // 2
            line0, data0, line1, data1, = \
                get_next(f0, None, None, f1, None, None, stamp_len, True, True)

            while line0 and line1:
                isects += 1

                if data0 == data1:
                    line0, data0, line1, data1 = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, True, True)

                elif data0 < data1:
                    line0, data0, line1, data1 = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, True, False)


                # data0 > data1
                else:
                    line0, data0, line1, data1 = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, False, True)


        return isects


    @staticmethod
    def numIsectSkipAhead(fn0, fn1):
        """ Compute the number of intersection attempts with skip-ahead
        intersection

        Parameters
        ----------

        fn0, fn1: str
            The filenames of the intersection traces

        Returns
        ------

        num_isects: int
            Number of intersection tests

        """
        def get_data(f, stamp_len):
            line = f.readline()
            if line:
                data = tuple(int(val) for val in line[:-1].split(",")[:-1])
            else:
                data = (float("inf"),)

            return line, data, data[stamp_len:-1]

        def get_next(f0, line0, data0, f1, line1, data1, stamp_len, advance0, advance1):
            new_fiber = False
            if data0:
                stamp0 = data0[stamp_len:-1]
            else:
                stamp0 = None

            if data1:
                stamp1 = data1[stamp_len:-1]
            else:
                stamp1 = None

            if advance0:
                old_stamp0 = stamp0
                line0, data0, stamp0 = get_data(f0, stamp_len)

                if old_stamp0 != stamp0:
                    new_fiber = True

            if advance1:
                old_stamp1 = stamp1
                line1, data1, stamp1 = get_data(f1, stamp_len)

                if old_stamp1 != stamp1:
                    new_fiber = True

            while line0 and line1:
                if stamp0 == stamp1:
                    break

                elif stamp0 < stamp1:
                    line0, data0, stamp0 = get_data(f0, stamp_len)
                    new_fiber = True

                # stamp0 > stamp1
                else:
                    line1, data1, stamp1 = get_data(f1, stamp_len)
                    new_fiber = True

            return line0, data0, line1, data1, new_fiber

        with open(fn0, "r") as f0, open(fn1, "r") as f1:
            # Throw away headers
            line0 = f0.readline()
            f1.readline()

            isects = 0
            curr = None

            stamp_len = (len(line0.split(",")) - 1) // 2
            line0, data0, line1, data1, _ = \
                get_next(f0, None, None, f1, None, None, stamp_len, True, True)

            while line0 and line1:
                # If both matched, there is nothing to skip
                if data0 == data1:
                    curr = None
                    isects += 1

                    line0, data0, line1, data1, new_fiber = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, True, True)

                # Intersect or skip tensor 0
                elif data0 < data1:
                    if curr != 0:
                        curr = 0
                        isects += 1

                    line0, data0, line1, data1, new_fiber = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, True, False)

                # Intersect or skip tensor 1
                # elif data0 > data1
                else:
                    if curr != 1:
                        curr = 1
                        isects += 1

                    line0, data0, line1, data1, new_fiber = \
                        get_next(f0, line0, data0, f1, line1, data1, stamp_len, False, True)

                if new_fiber:
                    curr = None

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
