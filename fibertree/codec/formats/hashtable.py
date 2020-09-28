from .compression_format import CompressionFormat
import sys 
# hash table per fiber

class HashTable(CompressionFormat):
    def __init__(self):
        self.name = "Hf"
        # if the hashtable length is fixed, don't need to write it as a payload
        CompressionFormat.__init__(self)
        self.hashtable_len = 8
        self.max_density = .8
        self.ht = [None] * self.hashtable_len
        self.ptrs = list()
        self.coords = list()
        self.payloads = list()
        
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
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor, shape=None):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth) 
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] == "Hf":
            cumulative_occupancy = (0, 0)
        occ_list = list()
        num_coords = len(a.getCoords())

        # init scratchpads
        # TODO: doubling
        # encode nonzeroes
        for ind, (val) in a:
            payload_to_add = None
            # add to payloads
            # if at the leaves, add the actual payloads
            if depth == len(ranks) - 1:
                payload_to_add = val.value
            # if in internal levels, also get the fiber
            else: 
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
                    
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                    
                if codec.fmts[depth + 1].encodeUpperPayload():
                    payload_to_add = cumulative_occupancy
                else:
                    payload_to_add = fiber_occupancy

            # add to HT
            self.insertElement(ind, payload=payload_to_add, count_stats=False)

            fiber_occupancy = fiber_occupancy + 1

        coords_key, payloads_key = codec.get_keys(ranks, depth)

        output[coords_key].extend(self.coords)
        output[payloads_key].extend(self.payloads)

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
        hash_key = self.get_hash_key(coord)
        bin_head = self.ht[hash_key]

        # assert bin_head != None
        # look for cached
        key = self.name + '_HT_' + str(hash_key)
        cached_val = self.cache.get(key)
        self.cache[key] = bin_head
        self.stats[self.ht_read_key] += 1
        print("\t{} coordToHandle: coord {}, hash_key {}".format(self.name, coord, hash_key))
        # search this bucket
        while bin_head != None:
            self.stats[self.coords_read_key] += 1 
            # print("\tbin head {}".format(bin_head))
            key = self.name + '_IdxToCoords_' + str(bin_head)
            cached_val = self.cache.get(key)
            self.cache[key] = self.coords[bin_head]

            # if found coord, return the pointer to it
            if self.coords[bin_head] == coord:
                return bin_head
            # advance pointer in bucket

            key = self.name + '_IdxToPtrs_' + str(bin_head)
            cached_val = self.cache.get(key)
            self.cache[key] = self.ptrs[bin_head]
            self.stats[self.ptrs_read_key] += 1

            bin_head = self.ptrs[bin_head]
        return None # not found
 
    # swoop API functions
    # given a coord, give a handle (same for coords, payloads) to it
    # TODO: if coord doesn't exist, return None? return next?
    def coordToHandleNoStats(self, coord):
        # encode coord
        hash_key = self.get_hash_key(coord)
        bin_head = self.ht[hash_key]

        assert bin_head != None
        # look for cached
        print("\t{} coordToHandle: coord {}, hash_key {}".format(self.name, coord, hash_key))
        # search this bucket
        while bin_head != None:
            if self.coords[bin_head] == coord:
                return bin_head
            # advance pointer in bucket
            bin_head = self.ptrs[bin_head]
        return None # not found
   
    # must return elts in sorted order on coord
    def setupSlice(self, base = 0, bound = None, max_num = None):
        super().setupSlice(base, bound, max_num)
        self.cur_handle = self.coordToHandle(base)
        
        if self.cur_handle == None: # not found
            val_at_min_handle = sys.maxsize
            min_handle = None

            # do a search through the coords to find the min greater than base
            for i in range(0, len(self.coords)):
                # look in the cache for it
                key = self.name + '_IdxToCoords_' + str(i)
                cached_val = self.cache.get(i)
                self.cache[key] = self.coords[i]

                self.stats[self.coords_read_key] += 1
                print("\tsearching coords: ind {}, coord {}, min_val {}".format(i, self.coords[i], val_at_min_handle))
                if min_handle == None:
                    if self.coords[i] > base:
                        min_handle = i
                        val_at_min_handle = self.coords[min_handle]
                else: 
                    assert min_handle != None
                    if self.coords[i] > base and self.coords[i] < val_at_min_handle:
                        min_handle = i
                        val_at_min_handle = self.coords[min_handle]

            self.cur_handle = min_handle        

        # print("\t{} setupSlice: curHandle = {}".format(self.name, self.cur_handle))

    # get hashtable key mod by table length
    def get_hash_key(self, val):
        return hash(str(val)) % self.hashtable_len
    
    # get next in iteration
    def nextInSlice(self):
        if self.cur_handle == None:
            return None
        if self.num_to_ret != None and self.num_to_ret < self.num_ret_so_far:
            return None
        if self.num_ret_so_far >= len(self.coords):
            return None
        cur_coord = self.coords[self.cur_handle]
        to_ret = self.cur_handle

        # look in the cache for it
        key = self.name + '_IdxToCoords_' + str(self.cur_handle)
        cached_val = self.cache.get(self.cur_handle)
        self.cache[key] = self.coords[self.cur_handle]
        self.stats[self.coords_read_key] += 1
        
        next_handle = None
        # need to do a linear pass to find the next coord in sorted order
        for i in range(0, len(self.coords)):
            key = self.name + '_IdxToCoords_' + str(i)
            cached_val = self.cache.get(i)
            self.cache[key] = self.coords[i]
     
            self.stats[self.coords_read_key] += 1
            if self.coords[i] > cur_coord:
                if next_handle == None or (self.coords[next_handle] > self.coords[i] and self.coords[i] > cur_coord):
                    next_handle = i
        self.cur_handle = next_handle
        return to_ret
    
    def double_table(self, count_stats):
        print("\t table doubling")
        # reset HT and ptrs
        self.hashtable_len = self.hashtable_len * 2
        self.ptrs = list()
        self.ht = [None] * self.hashtable_len
        for i in range(0, len(self.coords)):
            self.insertElement(self.coords[i], add_coord=False, count_stats=count_stats)
        assert(len(self.ptrs) == len(self.coords))
        # search for them all
        for i in range(0, len(self.coords)):
            assert self.coordToHandleNoStats(self.coords[i]) != None

    # modify coords, need to append 1 to payloads
    def insertElement(self, coord, payload=0, count_stats=True, add_coord=True):
        if coord == None:
            return None
            
        # encode coord
        hash_key = self.get_hash_key(coord)
        print("\tcoord: {}, hash key {}".format(coord, hash_key))
        bin_head = self.ht[hash_key]
        if count_stats:
            self.stats[self.ht_read_key] += 1
            key = self.name + '_HT_' + str(hash_key)
            cached_val = self.cache.get(key)
            self.cache[key] = bin_head

        # traverse this bucket
        while bin_head != None:
            # print("\tbin head {}".format(bin_head))
            if count_stats:
                key = self.name + '_IdxToCoords_' + str(bin_head)
                cached_val = self.cache.get(key)
                self.cache[key] = self.coords[bin_head]

            if self.coords[bin_head] == coord:
                # update payload or return because found
                return bin_head 
            bin_head = self.ptrs[bin_head]
        assert bin_head == None

        # make room for elt
        self.ptrs.append(self.ht[hash_key])
        self.ht[hash_key] = len(self.ptrs) - 1 
        # don't need to readd during doubling
        if add_coord:
            self.coords.append(coord)
            self.payloads.append(payload)
            self.stats[self.coords_write_key] += 1

        if count_stats:
            # add to stats
            self.stats[self.ht_write_key] += 1
            self.stats[self.ptrs_write_key] += 1

            # add payloads access to cache
            key = self.name + '_IdxToPayloads_' + str(len(self.coords) - 1)
            cached_val = self.cache.get(key)
            self.cache[key] = payload

        density = float(len(self.coords)) / self.hashtable_len
        if density >= self.max_density:
            self.double_table(count_stats)
        assert(len(self.coords) == len(self.payloads))
        return len(self.coords) - 1 # handle to coord == at the end

    def updatePayload(self, handle, payload):
        if handle == None:
            return None
        key = self.name + '_IdxToPayloads_' + str(handle)
        cached_val = self.cache.get(key)
        self.cache[key] = payload

        # update payload
        self.stats[self.payloads_write_key] += 1
        self.payloads[handle] = payload
        return handle

    # fiber handles must be reducible with int
    def getUpdatedFiberHandle(self):
        return len(self.payloads) + len(self.ht) 
        # return ((len(self.ht), len(self.payloads)), self)

    # default implementation == like in C
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
