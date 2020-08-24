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
        self.next_fmt = None
    # encode fiber into C format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        self.depth = depth

        # init vars
        fiber_occupancy = 0

        # TODO: HT to one payload
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] is "Hf":
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
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
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
        
        return fiber_occupancy
    
    #### fiber functions for AST

    # max length of slice
    def getSliceMaxLength(self):
        return len(self.coords)

    # return handle to existing coord that is at least coord
    def coordToHandle(self, coord):
        # print("\t{} coordToHandle for coord {}".format(self.name, coord))
        # if out of range, return None
        if len(self.coords) is 0:
            return None
        elif coord > self.coords[-1]:  
            # TODO: how to count cost out of range?add an access to append space to the end and look at the end
            return None
        elif coord <= self.coords[0]:
            return 0

        # if cached, incur no cost
        if self.prevCoordSearched is not None and self.prevCoordSearched == coord:
            return self.prevHandleAtCoordSearched
        # do a binary search if in range
        lo = 0
        hi = len(self.coords) - 1
        mid = 0
        # print("\t{} access before binary search {}".format(self.name, self.num_accesses))
        while lo <= hi:
            
            # print("\t coordToHandle: target {}, lo {}, mid {}, hi {}, reads {}".format(coord, lo, mid, hi, self.stats[self.coords_read_key]))
            self.stats[self.coords_read_key] += 1; # add to num accesses in binary search
            mid = math.ceil((hi + lo) / 2)
            # print("target {}, lo: {}, hi: {}, mid {}, coord {}".format(coord, lo, hi, mid, self.coords[mid]))
            if self.coords[mid] == coord:
                self.prevCoordSearched = coord
                self.prevHandleAtCoordSearched = mid
                return mid
            elif self.coords[mid] < coord:
                lo = mid + 1
            else: # self.coords[mid] > coord:
                hi = mid - 1
        # print()
        if (coord > self.coords[mid]):
            mid += 1
        self.prevCoordSearched = coord
        self.prevHandleAtCoordSearched = mid
        # print("\taccess after binary search {}".format(self.num_accesses))
        return mid

    # make space in coords and payloads for elt
    # return the handle
    def insertElement(self, coord):
        if coord is None:
            return None

        handle_to_add = self.coordToHandle(coord)
        
        # if went off the end 
        if handle_to_add is None:
            self.coords = self.coords + [coord]
            if self.is_leaf:
                self.payloads = self.payloads + [0]
            else:
                # assert(self.next_fmt is not None)
                self.payloads = self.payloads + [self.next_fmt()]
            self.stats[self.coords_write_key] += 1
            # NOTE: maybe charge for shifting payloads?
            # do we need to charge for allocating payload space at the end here? 
            # it will already be charged for the upate
            return len(self.coords) - 1

        # if adding a new coord, make room for it
        if self.coords[handle_to_add] is not coord:
            # add coord to coord list
            self.coords = self.coords[:handle_to_add] + [coord] + self.coords[handle_to_add:]

            # move payloads to make space
            if self.is_leaf:
                self.payloads = self.payloads[:handle_to_add] + [0] + self.payloads[handle_to_add:]
            else:
                self.payloads = self.payloads[:handle_to_add] + [self.next_fmt()] + self.payloads[handle_to_add:]

            # count number of accesses (number of elts shifted)
            self.stats[self.coords_write_key] += len(self.coords) - handle_to_add
            print("\t{} inserted coord {}".format(self.name, coord))
            self.printFiber()
        return handle_to_add

    # return handle for termination
    def updatePayload(self, handle, payload):
        if handle is None:
            return None
        
        if handle >= 0 and handle < len(self.payloads):
            # print(self.payloads)
            # print("setting payload at {} to {}".format(handle, payload))
            self.stats[self.payloads_write_key] += 1
            self.payloads[handle] = payload
        return handle

    def getUpdatedFiberHandle(self):
        # return update to occupancy and handle to internal python object
        return (len(self.coords), self)

    # print this fiber representation in C
    def printFiber(self):
        print("{} :: coords: {}, occupancies: {}, payloads: {}".format(self.name, self.coords, self.occupancies, self.payloads))
    
    # get size of representation
    def getSize(self): 
        # self.printFiber()
        assert(len(self.payloads) > 0)

        size = len(self.coords) + len(self.occupancies)
        # Don't need to store occupancies if lower level is U
        if not isinstance(self.payloads[0], CompressionFormat):
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
