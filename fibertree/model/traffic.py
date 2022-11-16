#cython: language_level=3
#cython: profile=True
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
    def filterTrace(input_fn, filter_fn, output_fn):
        """Filter a trace by keeping only accesses that occur at least once
        in the filter

        Parameters
        ----------

        input_fn: str
            Filename of the input trace

        filter_fn: str
            Filename of the filter trace

        output_fn: str
            Filename of the output trace
        """
        def get_data(line):
            if line:
                full = line[:-1].split(",")[:-1]
                return tuple(int(val) for val in full[len(full) // 2:])

            return ()

        with open(input_fn, "r") as f_in, open(filter_fn, "r") as f_fil, \
                open(output_fn, "w") as f_out:
            f_out.write(f_in.readline())
            f_fil.readline()

            line_in = f_in.readline()
            line_fil = f_fil.readline()

            data_in = get_data(line_in)
            data_fil = get_data(line_fil)[:len(data_in)]

            while line_in and line_fil:
                if data_in == data_fil:
                    f_out.write(line_in)

                    line_in = f_in.readline()
                    line_fil = f_fil.readline()

                    data_in = get_data(line_in)
                    data_fil = get_data(line_fil)[:len(data_in)]

                elif data_in < data_fil:
                    line_in = f_in.readline()
                    data_in = get_data(line_in)

                else:
                    line_fil = f_fil.readline()
                    data_fil = get_data(line_fil)[:len(data_in)]

    @staticmethod
    def _combineTraces(read_fn=None, write_fn=None, comb_fn=None):
        """Combine traces into a single trace"""
        assert comb_fn and (read_fn or write_fn)

        def next_line(f):
            line = f.readline()
            if line:
                split = line[:-1].split(",")
                return tuple(int(val) for val in split[:len(split) // 2]), line
            else:
                return (float("inf"),), line

        with open(comb_fn, "w") as f_comb:
            head = None
            f_read = None
            f_write = None

            # Read the first line
            if read_fn is not None:
                f_read = open(read_fn, "r")
                head = f_read.readline()
                read_line = next_line(f_read)
            else:
                read_line = (float("inf"),), ""

            if write_fn is not None:
                f_write = open(write_fn, "r")
                head = f_write.readline()
                write_line = next_line(f_write)
            else:
                write_line = (float("inf"),), ""

            f_comb.write(head[:-1] + ",is_write\n")

            # While either file has more
            while read_line[1] or write_line[1]:
                # If the write stamp is earlier
                if write_line[0] < read_line[0]:
                    f_comb.write(write_line[1][:-1] + ",True\n")
                    write_line = next_line(f_write)

                # Otherwise record the write
                else:
                    f_comb.write(read_line[1][:-1] + ",False\n")
                    read_line = next_line(f_read)

            if f_read:
                f_read.close()

            if f_write:
                f_write.close()

    @staticmethod
    def _buildPoint(split, mask, elems_per_line):
        """Build the access into a tensor for the form (coord, ... coord, pos),
        where (coord, ... coord) describes the fiber and pos describes the
        location within the fiber"""
        point = list(itertools.compress(split[(len(split) - 2) // 2:-2], mask))
        point[-1] = str(int(split[-2]) // elems_per_line * elems_per_line)
        return tuple(point)

    @staticmethod
    def _buildNextUseTrace(ranks, elems_per_line, input_fn, output_fn):
        """Build a trace of for each access to a tensor (as specified by its
        ranks), when the corresponding next use was"""
        out_split = os.path.splitext(output_fn)

        # Build a mask specifying the locations of the interesting ranks
        with open(input_fn) as f_in:
            head_in = f_in.readline()[:-1].split(",")
            iter_ranks = head_in[(len(head_in) - 2) // 2:-2]
            mask = [rank in ranks for rank in iter_ranks]

        # With each use, save the next use by iterating backwards
        last_points = {}
        with open(output_fn, "w") as f_out, FileReadBackwards(input_fn) as f_in:
            for line in f_in:
                # FileReadBackwards already removes the trailing newline
                split = line.split(",")

                # If we have reached the header, we are done
                if not split[0].isdigit():
                    break

                # Get the tensor point and compute its corresponding line
                point = Traffic._buildPoint(split, mask, elems_per_line)

                # If there is a last_use
                line_split = line.split(",")
                if point in last_points:
                    last_split = last_points[point].split(",")
                    new_csv = line + "," + last_points[point]
                else:
                    new_csv = line + "," + ",".join("None" for _ in split)

                f_out.write(new_csv + "\n")
                last_points[point] = line

            head_out = ",".join(head_in + [val + "_next" for val in head_in])
            f_out.write(head_out + "\n")

    @staticmethod
    def buffetTraffic(bindings, formats, trace_fns, capacity, line_sz, \
            loop_ranks=None):
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

        loop_ranks: Optional[Dict[str, str]]
            A map from the original rank to the rank it corresponds to in
            the loop order

        Note: assumes all fibers start at line boundaries and all elements
        reside on exactly one line (if the footprint is not a multiple of the
        line size, every line is padded)
        """
        def extract_binding(binding):
            return binding["tensor"], binding["rank"], binding["type"], binding["evict-on"]

        def pin_intermediate_writes(info):
            tensor, rank, type_, evict_on = info
            return rank != evict_on and (tensor, rank, type_, "write") in trace_fns

        def pre_sim_hook(bind_info):
            drain_info = {}
            ready_to_drain = {}

            # Prepare the drain queue info: for each key, what is the queue index
            # of the next element to be filled and the next element to be drained
            for tensor, rank, type_, _ in bind_info:
                drain_info[tensor, rank, type_] = [0, 0]
                ready_to_drain[tensor, rank, type_] = {}

            return drain_info, ready_to_drain

        def to_be_buffered(info, order, loop_ranks, trace, num_ranks):
            # Check if the next use is within the exploited reuse distance (ERD)
            if info[3] == "root":
                evict_end = 0
            else:
                evict_end = order.index(loop_ranks[info[3]]) + 1

            # Currently using the iteration stamp
            curr_stamp = trace[:evict_end]
            next_stamp = trace[num_ranks * 2 + 2:num_ranks * 2 + 2 + evict_end]

            return curr_stamp == next_stamp and trace[num_ranks * 2 + 2] is not None

        def add_elem(pre_sim_info, info, objs, obj, line_sz, occupancy,
                capacity, overflows):
            drain_info, ready_to_drain = pre_sim_info
            # Write-back info will be filled in later
            objs[info[0]][info[2]][obj] = [False, drain_info[info[:3]][0]]
            drain_info[info[:3]][0] += 1
            occupancy += line_sz

            # Track overflows
            if occupancy > capacity:
                overflows += 1

            pre_sim_info = drain_info, ready_to_drain
            return pre_sim_info, objs, occupancy, overflows

        def evict_elem(pre_sim_info, key, objs, obj, traffic, line_sz, occupancy):
            # Evict the element
            drain_info, ready_to_drain = pre_sim_info
            tensor, _, type_ = key
            ready_to_drain[key][objs[tensor][type_][obj][1]] = obj

            # Drain all available elements
            while drain_info[key][1] in ready_to_drain[key]:
                drain_obj = ready_to_drain[key][drain_info[key][1]]

                # If the line has been mutated and it needs to be saved, write it first
                if objs[tensor][type_][drain_obj][0]:
                    traffic[tensor]["write"] += line_sz

                del objs[tensor][type_][drain_obj]
                del ready_to_drain[key][drain_info[key][1]]
                drain_info[key][1] += 1
                occupancy -= line_sz

            pre_sim_info = drain_info, ready_to_drain
            return pre_sim_info, objs, traffic, occupancy

        return Traffic._bufferTraffic(bindings, formats, trace_fns, capacity,
            line_sz, loop_ranks, extract_binding, pin_intermediate_writes,
            pre_sim_hook, to_be_buffered, add_elem, evict_elem)

    @staticmethod
    def _bufferTraffic(bindings, formats, trace_fns, capacity, line_sz,
            loop_ranks, extract_binding, pin_intermediate_writes,
            pre_sim_hook, to_be_buffered, add_elem, evict_elem):
        """Generic buffer traffic function

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

        loop_ranks: Optional[Dict[str, str]]
            A map from the original rank to the rank it corresponds to in
            the loop order

        extract_binding: Callable[[Dict[str, int]], Tuple[int, ...]]
            extract_binding(binding) -> info
            A callback that extracts a tuple from the binding information

        pin_intermediate_writes: Callable[Tuple[int, ...], bool]
            pin_intermediate_writes(info) -> is_pinned
            A callback that determines whether intermediate writes should be pinned

        pre_sim_hook: Callable[[List[Tuple[int, ...]], Any]
            pre_sim_hook(bind_info) -> pre_sim_info
            A callback that does any other pre-buffer simulation processing necessary

        to_be_buffered: Callable[[Tuple[int, ...], List[str], Dict[str, str],
                Tuple[Optional[int], ...], int], bool]
            to_be_buffered(info, order, loop_ranks, trace, num_ranks) -> to_buffer
            A callback that determines whether the given access (as specificed
            by its trace) shoudl be buffered

        add_elem: Callable[[Any, Tuple[int, ...],
                    Dict[str, Dict[str, Dict[Tuple[int, ...], list]]],
                    Tuple[int, ...], int, int, int, int],
                Tuple[Any, Dict[str, Dict[str, Dict[Tuple[int, ...], list]]],
                    int, int]
            add_elem(pre_sim_info, info, objs, obj, line_sz, occupancy,
                capacity, overflows)
            A callback to add an element to the buffer

        evict_elem: Callable[[Any, Tuple[int, ...],
                    Dict[str, Dict[str, Dict[Tuple[int, ...], list]]],
                    Tuple[int, ...], Dict[str, Dict[str, int]], int, int]
                Tuple[Any, Dict[str, Dict[str, Dict[Tuple[int, ...], list]]],
                    Dict[str, Dict[str, int]], int]
            evict_elem(pre_sim_info, key, objs, obj, traffic, line_sz,
                occupancy) -> pre_sim_info, objs, traffic, occupancy
            A callback to evict an element

        Note: assumes all fibers start at line boundaries and all elements
        reside on exactly one line (if the footprint is not a multiple of the
        line size, every line is padded)
        """
        # Get the loop ranks of each tensor
        if loop_ranks is None:
            loop_ranks = {}

        loop_rank_ids = {}
        for tensor in formats:
            loop_rank_ids[tensor] = []
            for rank in formats[tensor].tensor.getRankIds():
                new_rank = rank
                if rank in loop_ranks:
                    new_rank = loop_ranks[rank]
                loop_rank_ids[tensor].append(new_rank)

        # First combine read and write traces
        read_write_traces = {}
        traffic = {}
        for (tensor, rank, type_, access), fn in trace_fns.items():
            # Initialize the traffic array
            if tensor not in traffic:
                traffic[tensor] = {}
            traffic[tensor][access] = 0

            # Make sure we have not combined the accesses yet
            key = tensor, rank, type_
            if key in read_write_traces:
                continue

            # Combine
            split_fn = os.path.splitext(fn)
            comb_fn = split_fn[0] + "-comb-" + "-".join(key) + split_fn[1]
            args = {access + "_fn": fn, "comb_fn": comb_fn}

            other_access = "read" if access == "write" else "write"
            if key + (other_access,) in trace_fns:
                args[other_access + "_fn"] = trace_fns[key + (other_access,)]

            Traffic._combineTraces(**args)
            read_write_traces[key] = comb_fn


        # Build traces with the next use
        next_use_traces = {}
        for key, fn in read_write_traces.items():
            tensor, rank, type_ = key
            elems_per_line = line_sz // formats[tensor].getElem(rank, type_)
            assert elems_per_line > 0

            split_fn = os.path.splitext(fn)
            next_fn = split_fn[0] + "-next-" + "-".join(key) + split_fn[1]

            Traffic._buildNextUseTrace(loop_rank_ids[tensor], elems_per_line, fn, next_fn)
            next_use_traces[key] = next_fn

        # Open all the traces
        traces = {}
        for key, fn in next_use_traces.items():
            traces[key] = FileReadBackwards(fn)

        # Get the loop order and pop off the headers
        order = []
        for file_ in traces.values():
            line = file_.readline()[:-1].split(",")

            start = (len(line) - 4) // 4
            if start > len(order):
                order = line[start:(start * 2)]

        # Fill the loop ranks
        for rank in order:
            loop_ranks[rank] = rank

        # Order the binding information, and get rid of the keys
        bind_info = [[] for _ in order]
        for binding in bindings:
            pos = order.index(loop_ranks[binding["rank"]])
            info = extract_binding(binding)
            tensor, rank, type_ = info[:3]

            # Make sure that the correct type is used
            assert (type_ == "elem" and formats[tensor].getLayout(rank) == "interleaved") \
                or (type_ == "coord" and formats[tensor].getLayout(rank) == "contiguous") \
                or (type_ == "payload" and formats[tensor].getLayout(rank) == "contiguous")

            bind_info[pos].append(info)

        # Flatten the binding info
        bind_info = [info for infos in bind_info for info in infos]

        # Compute index masks for each bound tensor
        masks = []
        for info in bind_info:
            tensor, rank = info[:2]
            end = order.index(loop_ranks[rank]) + 1
            masks.append(list(r in loop_rank_ids[tensor] for r in order[:end]))

        # Compute the number of elements per line for each binding
        elems_per_line = []
        for info in bind_info:
            tensor, rank, type_ = info[:3]
            footprint = formats[tensor].getElem(rank, type_)
            elems_per_line.append(line_sz // footprint)
            assert elems_per_line[-1] > 0

        # Compute the number of ranks listed in each file
        all_num_ranks = []
        for info in bind_info:
            all_num_ranks.append(order.index(loop_ranks[info[1]]) + 1)

        pre_sim_info = pre_sim_hook(bind_info)

        # Order the traces in the order they occur
        next_keys = []
        next_traces = []
        for i, info in enumerate(bind_info):
            next_key, next_trace = Traffic._extractNext(i, info, traces, order)
            next_keys.append(next_key)
            next_traces.append(next_trace)
        next_keys.sort()

        # Compute the last lines for all writable traces whose intermediate
        # writes we can ignore
        shapes = []
        for i, info in enumerate(bind_info):
            if pin_intermediate_writes(info):
                shape = formats[tensor].tensor.getShape(authoritative=True)
                assert shape is not None
                shapes.append(shape[formats[tensor].tensor.getRankIds().index(rank)])

            else:
                shapes.append(None)

        # Prepare the buffer
        objs = {}
        for tensor, rank, type_, _ in bind_info:
            if tensor not in objs:
                objs[tensor] = {}
            objs[tensor][type_] = {}

        # Simulate the buffet
        occupancy = 0
        overflows = 0

        # While at least one of the traces is still going
        while next_keys[0][:-1] != (float("inf"),) * len(order):
            i = next_keys[0][-1]
            key = bind_info[i][:3]
            tensor, rank, type_ = key

            # Get the tensor access
            trace = next_traces[i]
            num_ranks = all_num_ranks[i]
            point = list(itertools.compress(trace[num_ranks:num_ranks * 2], masks[i]))

            # We need to write this data if it was a write, and is not an
            # intermediate write that should never be written back
            is_write = trace[num_ranks * 2 + 1]
            write_back = is_write and (shapes[i] is None or trace[num_ranks * 2] < shapes[i])

            # Compute the corresponding line if the access is not "root"
            point[-1] = trace[num_ranks * 2] // elems_per_line[i] * elems_per_line[i]
            obj = tuple(point)

            # Record the access if needed
            new_traffic = obj not in objs[tensor][type_]
            if new_traffic and not is_write:
                traffic[tensor]["read"] += line_sz

            to_buffer = to_be_buffered(bind_info[i], order, loop_ranks, trace, num_ranks)
            if new_traffic:
                # Add an object only if it will be used within the ERD
                if to_buffer:
                    pre_sim_info, objs, occupancy, overflows = \
                        add_elem(pre_sim_info, bind_info[i], objs, obj, line_sz,
                            occupancy, capacity, overflows)
                    objs[tensor][type_][obj][0] = write_back

                # If new traffic and not buffered, add the write traffic
                elif write_back:
                    traffic[tensor]["write"] += line_sz

            # If the object will not be used again within the ERD, evict it
            elif not to_buffer:
                objs[tensor][type_][obj][0] = objs[tensor][type_][obj][0] or write_back
                pre_sim_info, objs, traffic, occupancy = \
                    evict_elem(pre_sim_info, key, objs, obj, traffic, line_sz, occupancy)

            # Otherwise, it was in the buffer and will be used again
            else:
                objs[tensor][type_][obj][0] = objs[tensor][type_][obj][0] or write_back

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
        for fn in read_write_traces.values():
            os.remove(fn)

        for fn in next_use_traces.values():
            os.remove(fn)

        return traffic, overflows

    @staticmethod
    def _extractNext(i, info, traces, order):
        """Get the next stamps for the given binding info"""
        # Get the trace
        line = traces[info[:3]].readline()

        # If there are no more lines, push this trace to the end
        if line == "":
            return (float("inf"),) * len(order) + (i,), []

        split = line[:-1].split(",")
        trace = []
        for val in split:
            if val.isdigit():
                trace.append(int(val))
            elif val == "None":
                trace.append(None)
            elif val == "True":
                trace.append(True)
            elif val == "False":
                trace.append(False)
            else:
                # Should never reach here
                raise ValueError("Unknown value: " + val)


        # Get the key that will be used to sort this trace
        # It is the iteration stamp padded with -1s and then the position of
        # the access type in the list of bindings
        num_ranks = len(trace) - 2
        key = [-1] * len(order) + [i]
        key[:num_ranks // 4] = trace[:num_ranks // 4]
        key = tuple(key)

        return key, trace

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
