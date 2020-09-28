from .compression_format import CompressionFormat
import sys
import math

# coordinate-payload list format (C)
class CoordinateList(CompressionFormat):
    def __init__(self):
        self.name = "C"
        CompressionFormat.__init__(self)
        # self.depth = None
        self.is_leaf = False
        # self.next_fmt = None
        # list of sizes of fibers so far in this rank
        self.occupancy_so_far = None

        # cache line locality
        self.elts_per_line = 4
    # encode fiber into C format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor, shape=None):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        self.depth = depth

        # init vars
        fiber_occupancy = 0

        # TODO: HT to one payload
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] == "Hf":
    	    cumulative_occupancy = [0, 0] 

        prev_nz = 0
        occ_list = list()
        if depth < len(ranks) - 1:
            self.next_fmt = codec.fmts[depth + 1]
        else:
            self.is_leaf = True
        for ind, (val) in a:
            # store coordinate explicitly
            coords = CoordinateList.encodeCoord(prev_nz, ind)

            # TODO: make the fiber rep an intermediate to YAML
            output[coords_key].extend(coords)
            self.coords.extend(coords)

            # keep track of nnz in this fiber
            fiber_occupancy = fiber_occupancy + 1

            # if at leaves, store payloads directly
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)
                self.payloads.append(val.value)
            else:
                # print("{}::set next fmt to {}".format(self.name, self.next_fmt))
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor,shape=shape)
                # keep track of actual occupancy (nnz in this fiber)
                
                # print("ind {}, depth{}, child {}, cumulative {}".format(ind, depth, child_occupancy, cumulative_occupancy))
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                # add cumulative or non-cumulative depending on settings
                codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)
                
                assert depth < len(ranks) - 1
                if codec.fmts[depth+1].encodeUpperPayload():
                    # TODO: make the choice for this to be cumulative
                    output[payloads_key].append(cumulative_occupancy)
                    self.occupancies.append(cumulative_occupancy)
                    self.payloads.append(fiber)

            prev_nz = ind + 1
        # print("{}:: coords {}, payloads {}".format(self.name, self.coords, self.payloads))
        self.fiber_occupancy = fiber_occupancy 
        return fiber_occupancy
    
    #### fiber functions for AST

    # max length of slice
    def getSliceMaxLength(self):
        return len(self.coords)

    # return handle to existing coord that is at least coord
    def coordToHandle(self, coord):
        # print("\t{} coordToHandle for coord {}".format(self.name, coord))
        # if out of range, return None
        if len(self.coords) == 0:
            return None
        
        elif coord > self.coords[-1]: # short path to end
            #  print("\tcoord searched off the end")
            key = self.name + "_handleToCoord_" + str(len(self.coords) - 1)
            print(key)
            print(self.cache)
            cached_val = self.cache.get(key)
            self.cache[key] = self.coords[-1]
            if self.name.startswith("Z"):
                print("{} coordToHandle coord {}, misses {}".format(self.name, coord, self.cache.miss_count))

            self.stats[self.coords_read_key] += 1; # add to num accesses in binary search
            return None
        elif coord <= self.coords[0]: # short path to beginning
            # print("\tcoord searched off the beginning")
            key = self.name + "_handleToCoord_0"
            cached_val = self.cache.get(key)
            self.cache[key] = self.coords[0]
            if self.name.startswith("Z"):
                print("{} coordToHandle coord {}, misses {}".format(self.name, coord, self.cache.miss_count))
            self.stats[self.coords_read_key] += 1; # add to num accesses in binary search
            return 0

        # do a binary search if in range
        lo = 0
        hi = len(self.coords) - 1
        mid = 0
        # print("\t{} access before binary search {}".format(self.name, self.num_accesses))
        while lo <= hi:
            # cache along the way in the binary search
            self.stats[self.coords_read_key] += 1; # add to num accesses in binary search

            # print("\t coordToHandle: target {}, lo {}, mid {}, hi {}, reads {}".format(coord, lo, mid, hi, self.stats[self.coords_read_key]))
            mid = math.ceil((hi + lo) / 2)
            coord_key = self.name + "_handleToCoord_" + str(mid)
            coord_at_mid = self.cache.get(coord_key)
            self.cache[coord_key] = self.coords[mid]
            # print("target {}, lo: {}, hi: {}, mid {}, coord {}".format(coord, lo, hi, mid, self.coords[mid]))
            if self.coords[mid] == coord:
                return mid
            elif self.coords[mid] < coord:
                lo = mid + 1
            else: # self.coords[mid] > coord:
                hi = mid - 1
        if (coord > self.coords[mid]):
            mid += 1
        # self.prevCoordSearched = coord
        # self.prevHandleAtCoordSearched = mid
        # print("\taccess after binary search {}".format(self.num_accesses))
        return mid

    # make space in coords and payloads for elt
    # return the handle
    def insertElement(self, coord):
        if coord == None:
            return None
        print("{} insertElt: coord {}, coords currently {}, misses before {}".format(self.name, coord, self.coords, self.cache.miss_count))

        handle_to_add = self.coordToHandle(coord)
        
        print("{} insertElt: coord {}, handle_to_add {}, misses before {}".format(self.name, coord, handle_to_add, self.cache.miss_count))
        # if went off the end 
        if handle_to_add == None:
            self.coords = self.coords + [coord]
            if self.is_leaf:
                self.payloads = self.payloads + [0]
            else:
                self.payloads = self.payloads + [self.next_fmt()]
            self.stats[self.coords_write_key] += 1
            handle = len(self.coords) - 1
            coords_key = self.name + "_handleToCoord_" + str(handle)
            payloads_key = self.name + "_handleToPayload_" + str(handle)

            print(self.cache)
            self.cache.get(coords_key)
            
            print("{} insertElt: coord {}, handle_to_add {}, misses after {}".format(self.name, coord, handle_to_add, self.cache.miss_count))
            self.cache.get(payloads_key)
            
            print("{} insertElt: coord {}, handle_to_add {}, misses after {}".format(self.name, coord, handle_to_add, self.cache.miss_count))
            self.cache[coords_key] = coord
            self.cache[payloads_key] = self.payloads[handle]
            print(self.cache)

            # fill out cache to end of line
            assert(len(self.payloads) == len(self.coords))
            end_of_line = self.round_up(handle, self.words_in_line)
            for i in range(handle, end_of_line):
                coords_key = self.name + "_handleToCoord_" + str(i)
                payloads_key = self.name + "_handleToPayload_" + str(i)
                if i < len(self.coords):
                    self.cache[coords_key] = self.coords[i]
                    self.cache[payloads_key] = self.payloads[i]
                else:
                    self.cache[coords_key] = 0
                    self.cache[payloads_key] = 0
            print(self.cache)
            return len(self.coords) - 1

        # if adding a new coord, make room for it
        if self.coords[handle_to_add] != coord:
            # add coord to coord list
            self.coords = self.coords[:handle_to_add] + [coord] + self.coords[handle_to_add:]

            # move payloads to make space
            if self.is_leaf:
                self.payloads = self.payloads[:handle_to_add] + [0] + self.payloads[handle_to_add:]
            else:
                self.payloads = self.payloads[:handle_to_add] + [self.next_fmt()] + self.payloads[handle_to_add:]

            # fill out cache to end of line
            assert(len(self.payloads) == len(self.coords))
            for i in range(handle_to_add, len(self.coords)):
                coords_key = self.name + "_handleToCoord_" + str(i)
                payloads_key = self.name + "_handleToPayload_" + str(i)
                cached_coord = self.cache.get(coords_key)

                self.cache[coords_key] = self.coords[i]
                self.cache[payloads_key] = self.payloads[i]
            if cached_coord == None: # DRAM miss
                    # bring the rest of the line in
                    end_of_line = self.round_up(i, self.words_in_line)
                    for j in range(i, end_of_line): 
                        coords_key = self.name + "_handleToCoord_" + str(j)
                        payloads_key = self.name + "_handleToPayload_" + str(j)
                        if j < len(self.coords):
                            self.cache[coords_key] = self.coords[j]
                            self.cache[payloads_key] = self.payloads[j]
                        else:
                            self.cache[coords_key] = 0
                            self.cache[payloads_key] = 0
   
            self.stats[self.coords_write_key] += len(self.coords) - handle_to_add
            # print("\t{} inserted coord {}".format(self.name, coord))
        return handle_to_add

    # API Methods
    def handleToPayload(self, handle):
        # if next level has implicit payloads above (e.g. U), payload is implicit
        if self.next_fmt != None and not self.next_fmt.encodeUpperPayload():
            print("\t\tnext level not encoded, ret {}".format(self.occupancy_so_far))
            
            return self.occupancy_so_far # self.idx_in_rank + handle
        return handle

    # API Methods
    def payloadToFiberHandle(self, payload):
        # if next level has implicit payloads above (e.g. U), payload is implicit
        if not self.next_fmt.encodeUpperPayload():
            print("\t{} next level not encoded, payload {} ret {}".format(self.name, payload, payload))
            return payload # self.idx_in_rank # + payload
        
        # print("\t{} payloadToFiberHandle:: ret {}".format(self.name, payload))
        return payload


    # return handle for termination
    def updatePayload(self, handle, payload):
        # print("\t{} updatePayload, handle = {}, payload = {}".format(self.name, handle, payload))
        if handle == None:
            return None
        
        if handle >= 0 and handle < len(self.payloads):
            # print(self.payloads)
            # print("setting payload at {} to {}".format(handle, payload))
            self.stats[self.payloads_write_key] += 1
            self.payloads[handle] = payload
        
        key = self.name + "_handleToPayload_" + str(handle)
        print("{} handleToPayload key: {}, miss count before {}".format(self.name, key, self.cache.miss_count))
        
        cached_val = self.cache.get(key)
        assert cached_val != None
        self.cache[key] = payload
        print(self.cache)
        print("{} handleToPayload key: {}, miss count after {}".format(self.name, key, self.cache.miss_count))

        return handle

    def getUpdatedFiberHandle(self):
        return len(self.coords)
    
    # print this fiber representation in C
    def printFiber(self):
        print("{} :: coords: {}, occupancies: {}, payloads: {}".format(self.name, self.coords, self.occupancies, self.payloads))
    
    # get size of representation
    def getSize(self): 
        # self.printFiber()
        if self.next_fmt != None and self.next_fmt.encodeUpperPayload():
            assert(len(self.payloads) > 0)

        size = len(self.coords) + len(self.occupancies)
        # Don't need to store occupancies if lower level is U
        # if not isinstance(self.payloads[0], CompressionFormat):
        size += len(self.payloads) 
        return size

    #### static methods

    # encode coord explicitly
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return [ind]

    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return True
