#cython: language_level=3
#cython: profile=False
"""Traffic

A class for computing the memory traffic incurred by a tensor
"""

import heapq

from fibertree import Tensor
from sortedcontainers import SortedList

class Traffic:
    """Class for computing the memory traffic of a tensor"""

    @staticmethod
    def buildTrace(rank, input_fn, output_fn, tensor=None, trace_type="iter", access_type="read"):
        """Compute a trace just for the given trace and filename

        Parameters
        ----------

        rank: str
            Rank whose trace to build

        input_fn: str
            Input trace filename

        output_fn: str
            Output trace filename

        tensor: Optional[int]
            Tensor number whose information to extract, None if irrelevant

        trace_type: str
            Type of the trace given; one of iter, intersect, or populate

        access_type: str
            Type of access to worry about; one of read or write

        """

        assert trace_type == "iter" or tensor is not None

        with open(input_fn, "r") as f_in, open(output_fn, "w") as f_out:
            head_in = f_in.readline()[:-1].split(",")

            # Find the start and end of the relevant names
            for start, head in enumerate(head_in):
                if not head.endswith("_pos"):
                    break

            end = head_in.index(rank) + 1

            other = None
            used = None
            if trace_type == "intersect":
                used = head_in.index(str(tensor) + "_match")

            elif trace_type == "populate":
                used = head_in.index(str(tensor) + "_access")

                if tensor % 2 == 0:
                    other = used + 1
                else:
                    other = used - 1

            f_out.write(",".join(head_in[start:end]) + "\n")

            # Build the trace, coalescing together consecutive accesses of the
            # same element
            last_stamp = None
            for line in f_in.readlines():
                split = line[:-1].split(",")

                # If not used continue
                if trace_type == "intersect" and split[used] == "False":
                    continue

                if trace_type == "populate" and split[used] == "False":
                    continue

                # If we only care about reads, don't track the writes on the output
                if trace_type == "populate" and access_type == "read" \
                        and tensor % 2 == 0 and split[other] == "False":
                    continue

                stamp = split[start:end]
                if stamp != last_stamp:
                    last_stamp = stamp
                    f_out.write(",".join(stamp) + "\n")


    @staticmethod
    def buffetTraffic(bindings, formats, traces, buffer_sz):
        """Compute the traffic loading data into this buffet

        Parameters
        ----------

        bindings: List[dict]
            A list of bindings to this buffet

        formats: Dict[str, Format]
            A dictionary from tensor names to their corresponding format objects

        traces: Dict[str, Dict[str, Dict[str, str]]]
            A nested dictionary of traces of the form {tensor: {rank: {type: trace_fn}}}

        """
        pass

    @staticmethod
    def buffetTraffic_old(prefix, tensor, rank, format_, mode="subtree"):
        """Compute the buffet traffic for a given tensor and rank

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        mode: str
            How much of the tensor to load in, either "subtree" or "fiber"

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory into the buffet
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)
        use_data = {}
        for use in uses:
            if use not in use_data.keys():
                if mode == "fiber":
                    footprint = format_.getFiber(*use)
                else:
                    footprint = format_.getSubTree(*use)

                use_data[use] = [footprint, 0]

            use_data[use][1] += 1

        return sum(data[0] * data[1] for data in use_data.values())

    @staticmethod
    def cacheTraffic_old(prefix, tensor, rank, format_, capacity):
        """Compute the cache traffic for given tensor and rank

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        capacity: int
            The capacity of the cache in bits

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory into the cache
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)

        # Save some state about the uses
        use_data = {}
        for i, use in enumerate(reversed(uses)):
            if use not in use_data:
                use_data[use] = [format_.getSubTree(*use), []]
            use_data[use][1].append(len(uses) - i - 1)

        # Model the cache
        objs = SortedList()

        occupancy = 0
        bits_loaded = 0

        for i, use in enumerate(uses):
            # If it is already in the cache, we incur no traffic
            if len(objs) > 0 and use == objs[0][1]:
                use_data[use][1].pop()
                objs.pop(0)

                if len(use_data[use][1]) == 0:
                    occupancy -= use_data[use][0]
                else:
                    objs.add((use_data[use][1][-1], use))

                continue

            # Size of the object
            size = use_data[use][0]

            # Evict until there is space in the cache
            while occupancy + size > capacity:
                obj = objs.pop(-1)
                occupancy -= use_data[obj[1]][0]

            # Now add in the new fiber
            bits_loaded += size

            # Immediately evict objects that will never be used again
            use_data[use][1].pop()
            if len(use_data[use][1]) > 0:
                objs.add((use_data[use][1][-1], use))
                occupancy += size

        return bits_loaded

    @staticmethod
    def lruTraffic_old(prefix, tensor, rank, format_, capacity):
        """Compute the cache traffic for given tensor and rank, given an LRU
        replacement policy

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        capacity: int
            The capacity of the cache in bits

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory into the cache
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)

        # Model the cache
        cache = []
        objs = {}
        use_data = {}

        occupancy = 0
        bits_loaded = 0

        for i, use in enumerate(uses):
            # If it is already in the cache, we incur no traffic
            if use in objs.keys() and objs[use] is not None:
                objs[use] = i
                heapq.heappush(cache, (i, use))
                continue

            # Size of the object
            if use not in use_data.keys():
                use_data[use] = format_.getSubTree(*use)
            size = use_data[use]

            # Evict until there is space in the cache
            while occupancy + size > capacity:
                stamp, obj = heapq.heappop(cache)
                # If we are actually evicting the most recent access, then
                # reduce the occupancy of the cache
                if objs[obj] == stamp:
                    objs[obj] = None
                    occupancy -= use_data[obj]

                # Otherwise, it can just be silently evicted (i.e. this object
                # wasn't actually in the cache in the first place)

            # Now add in the new fiber
            bits_loaded += size
            occupancy += size

            heapq.heappush(cache, (i, use))
            objs[use] = i

        return bits_loaded

    @staticmethod
    def streamTraffic_old(prefix, tensor, rank, format_):
        """Compute the traffic for streaming over a given tensor and rank

        WARNING: Should not be used for tensors iterated over with
        intersection

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory onto the chip
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)
        curr_fiber = None
        bits = format_.getRHBits(rank)
        fheader = format_.getFHBits(rank)
        elem = format_.getCBits(rank) + format_.getPBits(rank)

        for use in uses:
            fiber = use[:-1]
            if fiber != curr_fiber:
                bits += fheader
                curr_fiber = fiber
            bits += elem

        return bits

    @staticmethod
    def _getAllUses(prefix, tensor, rank):
        """
        Get an iterable of uses ordered by iteration stamp
        """
        with open(prefix + "-" + rank + "-iter.csv", "r") as f:
            cols = f.readline()[:-1].split(",")
            inds = [i for i, col in enumerate(cols) if col in tensor.getRankIds()]

            data = []
            for line in f.readlines():
                point = line[:-1].split(",")
                use = tuple(int(index) for i, index in enumerate(point) if i in inds)
                data.append(use)

        return data
