#cython: language_level=3
#cython: profile=False
"""Traffic

A class for computing the memory traffic incurred by a tensor
"""

import heapq
import itertools
import os

import bisect
from file_read_backwards import FileReadBackwards
from sortedcontainers import SortedList

from fibertree import Tensor

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
            for coord_start, head in enumerate(head_in):
                if not head.endswith("_pos"):
                    break

            coord_end = head_in.index(rank) + 1
            iter_end = head_in.index(rank + "_pos") + 1

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

            f_out.write(",".join(head_in[:iter_end] + head_in[coord_start:coord_end]) + "\n")

            # Build the trace, coalescing together consecutive accesses of the
            # same element
            last_stamp = None
            for line in f_in:

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

                stamp = split[:iter_end] + split[coord_start:coord_end]
                if stamp != last_stamp:
                    last_stamp = stamp
                    f_out.write(",".join(stamp) + "\n")
    @staticmethod
    def _buildPoint(split, mask, elems_per_line):
        """Build the access into a tensor"""
        point = list(itertools.compress(split[len(split) // 2:], mask))
        point[-1] = str(int(point[-1]) // elems_per_line * elems_per_line)
        return tuple(point)

    @staticmethod
    def _buildNextUseTrace(ranks, elems_per_line, input_fn, output_fn):
        """Build a trace of for each access to a tensor (as specified by its
        ranks), when the corresponding next use was"""

        out_split = os.path.splitext(output_fn)
        comb_fn = out_split[0] + "-coalesced" + out_split[1]
        rev_fn = out_split[0] + "-reversed" + out_split[1]

        # Build a mask specifying the locations of the interesting ranks
        with open(input_fn) as f_in:
            head_in = f_in.readline()[:-1].split(",")
            iter_ranks = head_in[len(head_in) // 2:]
            mask = [rank in ranks for rank in iter_ranks]

        # First coalesce together accesses onto a single line
        last_point = None
        with open(comb_fn, "w") as f_comb, open(input_fn, "r") as f_in:
            f_comb.write(f_in.readline())
            for line in f_in:
                split = line[:-1].split(",")

                # Get the tensor point and compute its corresponding line
                point = Traffic._buildPoint(split, mask, elems_per_line)
                if last_point != point:
                    f_comb.write(line)

                last_point = point

        # With each use, save the next use by iterating backwards
        last_points = {}
        with open(rev_fn, "w") as f_rev, FileReadBackwards(comb_fn) as f_comp:
           for line in f_comp:
                # FileReadBackwards already removes the trailing newline
                split = line.split(",")

                # If we have reached the header, we are done
                if not split[0].isdigit():
                    break

                # Get the tensor point and compute its corresponding line
                point = Traffic._buildPoint(split, mask, elems_per_line)

                # If there is a last_use
                if point in last_points:
                    new_csv = line + "," + last_points[point]
                else:
                    new_csv = line + "," + ",".join("None" for _ in split)

                f_rev.write(new_csv + "\n")
                last_points[point] = line

        # Now reverse the file to make the output
        with open(output_fn, "w") as f_out, FileReadBackwards(rev_fn) as f_rev:
            # First write the header
            head_out = ",".join(head_in + [val + "_next" for val in head_in])
            f_out.write(head_out + "\n")

            # Write the trace
            for line in f_rev:
                f_out.write(line + "\n")

        os.remove(comb_fn)
        os.remove(rev_fn)


    @staticmethod
    def buffetTraffic(bindings, formats, trace_fns, capacity, line_sz):
        """Compute the traffic loading data into this buffet

        Parameters
        ----------

        bindings: List[dict]
            A list of the binding information

        formats: Dict[str, Format]
            A dictionary from tensor names to their corresponding format objects

        trace_fns: Dict[Tuple[str, str, str, str], str]]]
            A nested dictionary of traces of the form
            {(tensor, rank, type, access): trace_fn}}}
            where type is one of "elem", "coord", or "payload" and access is
            "read" or "write"

        capacity: int
            The number of bits that fit in the buffet

        line_sz: int
            The number of bits across which spatial locality is exploited
            (e.g., buffer line size)

        Note: assumes all fibers start at line boundaries and all elements
        reside on exactly one line (if the footprint is not a multiple of the
        line size, every line is padded)
        """

        # Build traces with the next use
        next_use_traces = {}
        for key, fn in trace_fns.items():
            tensor, rank, type_, access = key
            rank_ids = formats[tensor].tensor.getRankIds()
            elems_per_line = line_sz // formats[tensor].getElem(rank, type_)
            assert elems_per_line > 0

            split_fn = os.path.splitext(fn)
            next_fn = split_fn[0] + "-next-" + "-".join(key) + split_fn[1]

            Traffic._buildNextUseTrace(rank_ids, elems_per_line, fn, next_fn)
            next_use_traces[key] = next_fn

        # Open all the traces
        traces = {}
        for key, fn in next_use_traces.items():
            traces[key] = open(fn, "r")

        # Get the loop order and pop off the headers
        order = []
        for file_ in traces.values():
            line = file_.readline()[:-1].split(",")

            start = len(line) // 4
            if start > len(order):
                order = line[start:(start * 2)]

        # Order the binding information, and get rid of the keys
        bind_info = [[] for _ in order]
        for binding in bindings:
            pos = order.index(binding["rank"])
            tensor, rank, type_, evict_on = \
                binding["tensor"], binding["rank"], binding["type"], binding["evict-on"]

            # Make sure that the correct type is used
            assert (type_ == "elem" and formats[tensor].getLayout(rank) == "interleaved") \
                or (type_ == "coord" and formats[tensor].getLayout(rank) == "contiguous") \
                or (type_ == "payload" and formats[tensor].getLayout(rank) == "contiguous")

            # If we have a trace for the read of this binding
            if (tensor, rank, type_, "read") in traces:
                bind_info[pos].append((tensor, rank, type_, "read", evict_on))

            # If we have a trace for the write of this binding
            if (tensor, rank, type_, "write") in traces:
                bind_info[pos].append((tensor, rank, type_, "write", evict_on))

        # Flatten the binding info
        bind_info = [info for infos in bind_info for info in infos]

        # Compute index masks for each bound tensor
        masks = []
        for info in bind_info:
            end = order.index(info[1]) + 1
            ranks = formats[info[0]].tensor.getRankIds()
            mask = [rank in ranks for rank in order[:end]]

        # Compute the number of elements per line for each binding
        elems_per_line = []
        for tensor, rank, type_, _, _ in bind_info:
            footprint = formats[tensor].getElem(rank, type_)
            elems_per_line.append(line_sz // footprint)
            assert elems_per_line[-1] > 1

        # Order the traces in the order they occur
        next_keys = []
        next_traces = []
        for i, info in enumerate(bind_info):
            next_key, next_trace = Traffic._extractNext(i, info, traces, order)
            next_keys.append(next_key)
            next_traces.append(next_trace)
        next_keys.sort()

        # Simulate the buffet
        objs = set()
        occupancy = 0
        traffic = 0
        overflows = 0

        # While at least one of the traces is still going
        while next_keys[0][:-1] != (float("inf"),) * len(order):
            i = next_keys[0][-1]
            tensor, rank, type_, access, evict_on = bind_info[i]

            # Get the tensor access
            trace = next_traces[i]
            access = list(itertools.compress(trace[len(trace) // 4:len(trace) // 2], mask))

            # Compute the corresponding line
            access[-1] = access[-1] // elems_per_line[i] * elems_per_line[i]
            obj = (tensor, type_, tuple(access))

            # Record the access if needed
            new_traffic = obj not in objs
            if new_traffic:
                traffic += line_sz

            # Check if the next use is within the exploited reuse distance (ERD)
            if evict_on == "root":
                evict_end = 0
            else:
                evict_end = order.index(evict_on) + 1

            # Currently using the iteration stamp
            curr_stamp = trace[:evict_end]
            next_stamp = trace[len(trace) // 2:len(trace) // 2 + evict_end]

            # Add an object only if it will be used within the ERD
            if curr_stamp == next_stamp and new_traffic \
                    and trace[len(trace) // 2] is not None:
                objs.add(obj)
                occupancy += line_sz

                # Track overflows
                if occupancy > capacity:
                    overflows += 1

            # If the object will not be used again within the ERD, evict it
            elif curr_stamp != next_stamp and not new_traffic:
                objs.remove(obj)
                occupancy -= line_sz

            # Advance the relevant trace
            del next_keys[0]
            next_key, next_trace = Traffic._extractNext(i, bind_info[i], traces, order)
            next_traces[i] = next_trace

            j = bisect.bisect_left(next_keys, next_key)
            next_keys.insert(j, next_key)

        # Close all files
        for file_ in traces.values():
            file_.close()

        # Remove all of the newly created files
        for fn in next_use_traces.values():
            os.remove(fn)

        return traffic, overflows

    @staticmethod
    def _extractNext(i, info, traces, order):
        """Get the next stamps for the given binding info"""
        # Get the trace
        line = traces[info[:4]].readline()

        # If there are no more lines, push this trace to the end
        if line == "":
            return (float("inf"),) * len(order) + (i,), []

        split = line[:-1].split(",")
        trace = [None if val == "None" else int(val) for val in split]

        # Get the key that will be used to sort this trace
        # It is the iteration stamp padded with -1s and then the position of
        # the access type in the list of bindings
        key = [-1] * len(order) + [i]
        key[:len(trace) // 4] = trace[:len(trace) // 4]
        key = tuple(key)

        return key, trace

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
