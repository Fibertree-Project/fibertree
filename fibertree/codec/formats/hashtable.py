from .compression_format import CompressionFormat
import sys 
# hash table per fiber

class HashTable(CompressionFormat):
    def __init__(self):
        self.name = "Hf"
        # if the hashtable length is fixed, don't need to write it as a payload
        CompressionFormat.__init__(self)
        self.hashtable_len = 6
        self.ht_read_key = "num_ht_reads"
        self.ht_write_key = "num_ht_writes"
        self.ptrs_read_key = "num_ptrs_reads"
        self.ptrs_write_key = "num_ptrs_writes"
        self.stats[self.ht_read_key] = 0 
        self.stats[self.ht_write_key] = 0
        self.stats[self.ptrs_read_key] = 0
        self.stats[self.ptrs_write_key] = 0

    @staticmethod
    def helper_add(output, key, to_add):
        if key in output:
            output[key].extend(to_add)
        else:
            output[key] = to_add

    # encode fiber in H format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth) 
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] is "Hf":
            cumulative_occupancy = (0, 0)
        occ_list = list()
        num_coords = len(a.getCoords())

        # init scratchpads
        # TODO: doubling
        self.ht = [None] * self.hashtable_len
        self.ptrs = list()
        self.coords = list()
        self.payloads = list()

        # encode nonzeroes
        for ind, (val) in a:
            # TODO: make this a function
            # cumulative_occupancy = cumulative_occupancy + child_occupancy


            # add to payloads
            # if at the leaves, add the actual payloads
            if depth == len(ranks) - 1:
                self.payloads.append(val.value)
            # if in internal levels, also get the fiber
            else: 
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
                    
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                    
                if codec.fmts[depth + 1].encodeUpperPayload():
                    self.payloads.append(cumulative_occupancy)
                else:
                    self.payloads.append(fiber_occupancy)
            
            # encode coord
            hash_key = ind % self.hashtable_len
            # print("ind: {}, hash key {}".format(ind, hash_key))
            bin_head = self.ht[hash_key]
            while bin_head is not None:
                # print("\tbin head {}".format(bin_head))
                if self.coords[bin_head] is ind:
                    # update payload or return because found
                    self.payloads[bin_head] = (val)
                    return True
                bin_head = self.ptrs[bin_head]

            self.ptrs.append(self.ht[hash_key])
            self.ht[hash_key] = len(self.ptrs) - 1 
            self.coords.append(ind)
            assert(len(self.ptrs) == len(self.coords))
            
            fiber_occupancy = fiber_occupancy + 1

        coords_key, payloads_key = codec.get_keys(ranks, depth)
        # ptrs_key = "ptrs_{}".format(ranks[depth].lower())
        # ht_key = "ht_{}".format(ranks[depth].lower())

        output[coords_key].extend(self.coords)
        output[payloads_key].extend(self.payloads)
        # output[ptrs_key].extend(self.ptrs)
        # output[ht_key].extend(self.ht)

        # linearize output dict
        # coords in the format of two lists: 
        # 1. like the segment table in CSR, that points to the start of what was in that bucket
        # 2. linearization of buckets in contiguous order

        total_size = self.hashtable_len + len(self.ptrs) + len(self.coords) + len(self.payloads)
        return [fiber_occupancy, self.hashtable_len]

    # TODO: fillin
    def getSize(self):
        return self.hashtable_len + len(self.ptrs) + len(self.coords) + len(self.payloads)

    def printFiber(self):
        print("{} :: ht: {}, ptrs {}, coords {}, payloads {}".format(self.name, self.ht, self.ptrs, self.coords, self.payloads))

    # swoop API functions
    # given a coord, give a handle (same for coords, payloads) to it
    # TODO: if coord doesn't exist, return None? return next?
    def coordToHandle(self, coord):
        # encode coord
        hash_key = coord % self.hashtable_len
        bin_head = self.ht[hash_key]
        self.stats[self.ht_read_key] += 1
        print("{} coordToHandle: coord {}, hash_key {}".format(self.name, coord, hash_key))
        while bin_head is not None:
            # print("\tbin head {}".format(bin_head))
            if self.coords[bin_head] is coord:
                self.stats[self.coords_read_key] += 1 
                # update payload or return because found
                return bin_head
            bin_head = self.ptrs[bin_head]
            self.stats[self.ptrs_read_key] += 1

    
    # must return elts in sorted order on coord
    def setupSlice(self, base = 0, bound = None, max_num = None):
        super().setupSlice(base, bound, max_num)
        self.cur_handle = self.coordToHandle(base)
        
        if self.cur_handle is None: # not found
            val_at_min_handle = sys.maxsize
            min_handle = None

            # do a search through the coords to find the min greater than base
            for i in range(0, len(self.coords)):
                self.stats[self.coords_read_key] += 1
                print("\tsearching coords: ind {}, coord {}, min_val {}".format(i, self.coords[i], val_at_min_handle))
                if min_handle is None:
                    if self.coords[i] > base:
                        min_handle = i
                        val_at_min_handle = self.coords[min_handle]
                else: 
                    assert min_handle is not None
                    if self.coords[i] > base and self.coords[i] < val_at_min_handle:
                        min_handle = i
                        val_at_min_handle = self.coords[min_handle]

            self.cur_handle = min_handle        

        # print("\t{} setupSlice: curHandle = {}".format(self.name, self.cur_handle))
            
    # get next in iteration
    def nextInSlice(self):
        if self.cur_handle is None:
            return None
        if self.num_to_ret is not None and self.num_to_ret < self.num_ret_so_far:
            return None
        if self.num_ret_so_far >= len(self.coords):
            return None
        cur_coord = self.coords[self.cur_handle]
        self.stats[self.coords_read_key] += 1
        to_ret = self.cur_handle # cur_coord
        next_handle = None
        # need to do a linear pass to find the next coord in sorted order
        for i in range(0, len(self.coords)):
            self.stats[self.coords_read_key] += 1
            if self.coords[i] > cur_coord:
                if next_handle is None or (self.coords[next_handle] > self.coords[i] and self.coords[i] > cur_coord):
                    next_handle = i
        # if next_handle is not None:
            # print("\tnext handle {}, coord at handle {}".format(next_handle, self.coords[next_handle]))
        self.cur_handle = next_handle
        # print("\treturning {}".format(to_ret))
        return to_ret

    # modify coords, need to append 1 to payloads
    def insertElement(self, coord):
        if coord is None:
            return None
            
        # encode coord
        hash_key = coord % self.hashtable_len
        # print("ind: {}, hash key {}".format(ind, hash_key))
        bin_head = self.ht[hash_key]
        # TODO: if insertElt goes into building the fiber, we only want to count this
        # later update
        self.stats[self.ht_read_key] += 1
        while bin_head is not None:
            # print("\tbin head {}".format(bin_head))
            if self.coords[bin_head] is coord:
                # update payload or return because found
                return bin_head 
            bin_head = ptrs[bin_head]

        self.ptrs.append(self.ht[hash_key])
        self.ht[hash_key] = len(self.ptrs) - 1 
        self.coords.append(coord)
        
        # add to stats
        self.stats[self.ht_write_key] += 1
        self.stats[self.ptrs_write_key] += 1
        self.stats[self.coords_write_key] += 1

        assert(len(self.ptrs) == len(self.coords))

        self.payloads.append(0)
        assert(len(self.coords) == len(self.payloads))
        return len(self.coords) - 1 # handle to coord is at the end

    def updatePayload(self, handle, payload):
        if handle is None:
            return None
        # update payload
        self.stats[self.payloads_write_key] += 1
        self.payloads[handle] = payload

        return handle

    def getUpdatedFiberHandle(self):
        return ((len(self.ht), len(self.payloads)), self)

    # default implementation is like in C
    # overwrite if this is changed
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        output = list()
        for i in range(prev_ind, ind):
            output.append(0)
        output.append(payload)
        return output

    @staticmethod
    def endPayloads(num_to_pad):
        return []

    # implicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return True
    
    @staticmethod 
    def startOccupancy():
        return [0, 0]
