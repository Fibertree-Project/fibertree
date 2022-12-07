#cython: language_level=3
#cython: profile=True
"""Traffic

A class for computing the memory traffic incurred by a tensor
"""

import functools
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
    def buffetTraffic(bindings, formats, trace_fns, capacity, line_sz,
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

        def to_be_buffered(bind_info, bind_pos, capacity, loop_ranks,
                num_ranks, obj, objs, occupancy, order, shapes, sim_info, trace):
            # Check if the next use is within the exploited reuse distance (ERD)
            evict_on = bind_info[bind_pos][3]
            if evict_on == "root":
                evict_end = 0
            else:
                evict_end = order.index(loop_ranks[evict_on]) + 1

            # Currently using the iteration stamp
            curr_stamp = trace[:evict_end]
            next_stamp = trace[num_ranks * 2 + 2:num_ranks * 2 + 2 + evict_end]

            return curr_stamp == next_stamp and trace[num_ranks * 2 + 2] is not None, sim_info

        def add_elem(bind_info, bind_pos, capacity, line_sz, num_ranks, obj, objs,
                occupancy, overflows, shapes, sim_info, trace, traffic):
            tensor, rank, type_, _ = bind_info[bind_pos]
            key = tensor, rank, type_
            drain_info, ready_to_drain = sim_info
            # Write-back info will be filled in later
            objs[tensor][type_][obj] = [False, drain_info[key][0]]
            drain_info[key][0] += 1
            occupancy += line_sz

            # Track overflows
            if occupancy > capacity:
                overflows += 1

            sim_info = drain_info, ready_to_drain
            return objs, occupancy, overflows, sim_info, traffic

        def evict_elem(key, line_sz, obj, objs, occupancy, sim_info, traffic):
            # Evict the element
            drain_info, ready_to_drain = sim_info
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

            sim_info = drain_info, ready_to_drain
            return objs, occupancy, sim_info, traffic

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

        extract_binding: Callable
            extract_binding(binding) -> info

        pin_intermediate_writes: Callable
            pin_intermediate_writes(info) -> is_pinned

        pre_sim_hook: Callable
            pre_sim_hook(bind_info) -> sim_info

        to_be_buffered: Callable
            to_be_buffered(bind_info, bind_pos, capacity, loop_ranks,
                    num_ranks, obj, objs, occupancy, order, shapes, sim_info,
                    trace) ->
                to_buffer, sim_info

        add_elem: Callable
            add_elem(bind_info, bind_pos, capacity, line_sz, num_ranks, obj,
                    objs, occupancy, overflows, shapes, sim_info, trace,
                    traffic) ->
               objs, occupancy, overflows, sim_info, traffic

        evict_elem: Callable
            evict_elem(key, line_sz, obj, objs, occupancy, sim_info, traffic) ->
                objs, occupancy, sim_info, traffic

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

        sim_info = pre_sim_hook(bind_info)

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
        for info in bind_info:
            tensor = info[0]
            type_ = info[2]
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

            to_buffer, sim_info = to_be_buffered(bind_info, i, capacity,
                loop_ranks, num_ranks, obj, objs, occupancy, order, shapes,
                sim_info, trace)
            if new_traffic:
                # Add an object only if it will be used within the ERD
                if to_buffer:
                    objs, occupancy, overflows, sim_info, traffic = \
                        add_elem(bind_info, i, capacity, line_sz, num_ranks,
                            obj, objs, occupancy, overflows, shapes, sim_info,
                            trace, traffic)
                    objs[tensor][type_][obj][0] = write_back

                # If new traffic and not buffered, add the write traffic
                elif write_back:
                    traffic[tensor]["write"] += line_sz

            # If the object will not be used again within the ERD, evict it
            elif not to_buffer:
                objs[tensor][type_][obj][0] = objs[tensor][type_][obj][0] or write_back
                objs, occupancy, sim_info, traffic = \
                    evict_elem(key, line_sz, obj, objs, occupancy, sim_info,
                        traffic)

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
    def cacheTraffic(bindings, formats, trace_fns, capacity, line_sz,
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
        @functools.total_ordering
        class ListElem:
            def __init__(self, bind_pos, next_access, obj):
                self.next_access = next_access
                self.obj = obj
                self.pos = bind_pos

            def __eq__(self, other):
                return self.next_access == other.next_access and self.pos == other.pos

            def __lt__(self, other):
                if self.next_access != other.next_access:
                    return self.next_access < other.next_access

                # Fall back on the position if the next access is at the same point
                return self.pos < other.pos

            def __repr__(self):
                return str((self.next_access, self.obj, self.pos))

        def extract_binding(binding):
            return binding["tensor"], binding["rank"], binding["type"]

        def pin_intermediate_writes(info):
            return info + ("write",) in trace_fns

        def pre_sim_hook(bind_info):
            next_evict = SortedList()

            pinned = {}
            for tensor, _, type_ in bind_info:
                if tensor not in pinned:
                    pinned[tensor] = {}

                if type_ not in pinned[tensor]:
                    pinned[tensor][type_] = set()

            return next_evict, pinned, None

        def to_be_buffered(bind_info, bind_pos, capacity, loop_ranks,
                num_ranks, obj, objs, occupancy, order, shapes, sim_info, trace):
            next_evict, pinned, _ = sim_info
            next_access = trace[num_ranks * 2 + 2:num_ranks * 3 + 2]
            tensor, _, type_ = bind_info[bind_pos]

            # Do not buffer if never used again
            if trace[num_ranks * 2 + 2] is None:
                if obj in objs[tensor][type_]:
                    if next_evict and next_evict[0].obj == obj:
                        list_elem = next_evict.pop(0)
                    else:
                        assert obj in pinned[tensor][type_]
                        list_elem = objs[tensor][type_][obj][1]

                    list_elem.next_access = [float("inf")]

                else:
                    list_elem = None

                sim_info = next_evict, pinned, list_elem
                return False, sim_info

            # If this element is in the cache, but not pinned
            if next_evict and next_evict[0].obj == obj \
                    and next_evict[0].pos == bind_pos:
                list_elem = next_evict.pop(0)
                list_elem.next_access = next_access

                next_evict.add(list_elem)
                sim_info = next_evict, pinned, list_elem

                return True, sim_info

            # If the element is pinned
            elif obj in pinned[tensor][type_]:
                list_elem = objs[tensor][type_][obj][1]
                list_elem.next_access = next_access

                sim_info = next_evict, pinned, list_elem
                return True, sim_info

            assert obj not in objs[tensor][type_]

            list_elem = ListElem(bind_pos, next_access, obj)
            sim_info = next_evict, pinned, list_elem
            # Definitely buffer if there is space in the buffer
            if occupancy + line_sz <= capacity:
                to_buffer = True

            # Definitely buffer if this is a pinned element
            elif shapes[bind_pos] is not None and trace[num_ranks * 2] >= shapes[bind_pos]:
                to_buffer = True

            # Do not buffer if we are full of pinned elements
            elif not next_evict:
                to_buffer = False

            # Otherwise, make sure the next thing to evict is not the element
            # we would have buffered
            else:
                to_buffer = list_elem <= next_evict[-1]

            return to_buffer, sim_info

        def add_elem(bind_info, bind_pos, capacity, line_sz, num_ranks, obj, objs,
                occupancy, overflows, shapes, sim_info, trace, traffic):

            next_evict, pinned, list_elem = sim_info
            # Evict if necessary to make space
            while occupancy + line_sz > capacity:
                if next_evict:
                    evict_elem = next_evict.pop(-1)

                    # If the line has been mutated and it needs to be saved, write it first
                    evict_tensor, _, evict_type = bind_info[evict_elem.pos]
                    if objs[evict_tensor][evict_type][evict_elem.obj][0]:
                        traffic[evict_tensor]["write"] += line_sz

                    # Evict and drain the element
                    del objs[evict_tensor][evict_type][evict_elem.obj]

                    occupancy -= line_sz

                # The pinned data has filled the cache
                else:
                    overflows += 1
                    break

            # If this element is not pinned, add it to the heap
            tensor, _, type_ = bind_info[bind_pos]
            if shapes[bind_pos] is None or trace[num_ranks * 2] < shapes[bind_pos]:
                next_evict.add(list_elem)

            # Otherwise pin the element
            else:
                pinned[tensor][type_].add(obj)

            objs[tensor][type_][obj] = [False, list_elem]
            occupancy += line_sz

            sim_info = next_evict, pinned, None
            return objs, occupancy, overflows, sim_info, traffic

        def evict_elem(key, line_sz, obj, objs, occupancy, sim_info, traffic):
            next_evict, pinned, list_elem = sim_info
            tensor, _, type_ = key

            # Ensure that there has been no error
            assert list_elem.next_access == [float("inf")]

            # If the line has been mutated and it needs to be saved, write it first
            if objs[tensor][type_][list_elem.obj][0]:
                traffic[tensor]["write"] += line_sz

            # Remove the object from objs and pinned if it is there
            del objs[tensor][type_][obj]
            if obj in pinned[tensor][type_]:
                pinned[tensor][type_].remove(obj)

            occupancy -= line_sz

            sim_info = next_evict, pinned, None

            return objs, occupancy, sim_info, traffic

        return Traffic._bufferTraffic(bindings, formats, trace_fns, capacity,
            line_sz, loop_ranks, extract_binding, pin_intermediate_writes,
            pre_sim_hook, to_be_buffered, add_elem, evict_elem)
