from .compression_format import CompressionFormat
import sys
import math

# coordinate-payload list format (C)
class CoordinateList(CompressionFormat):
    def __init__(self):
        self.name = "C"
        CompressionFormat.__init__(self)

    # @staticmethod
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        
        # init vars
        fiber_occupancy = 0

        # TODO: HT to one payload
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] is "Hf":
    	    cumulative_occupancy = [0, 0] 

        occ_list = list()
        # occ_list.append(cumulative_occupancy)
        prev_nz = 0
        
        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
            # keep track of actual occupancy (nnz in this fiber)
            
            # print("ind {}, depth{}, child {}, cumulative {}".format(ind, depth, child_occupancy, cumulative_occupancy))
            if isinstance(cumulative_occupancy, int):
                cumulative_occupancy = cumulative_occupancy + child_occupancy
            else:
                cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
            # add cumulative or non-cumulative depending on settings
            codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)
            
            # store coordinate explicitly
            coords = CoordinateList.encodeCoord(prev_nz, ind)

            # TODO: make the fiber rep an intermdiate to YAML
            output[coords_key].extend(coords)
            self.coords.extend(coords)

            # keep track of nnz in this fiber
            fiber_occupancy = fiber_occupancy + 1

            # if at leaves, store payloads directly
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)
                self.payloads.append(val.value)

            prev_nz = ind + 1
        
        # explicit payloads for next level
        if depth < len(ranks) - 1 and codec.fmts[depth+1].encodeUpperPayload():
            output[payloads_key].extend(occ_list)
            self.payloads.extend(occ_list)
        return fiber_occupancy, occ_list
    
    #### fiber functions for AST

    # max length of slice
    def getSliceMaxLength(self):
        return len(self.coords)

    # return handle to existing coord that is at least coord
    def coordToHandle(self, coord):
        # if out of range, return None
        # print(coord)
        # print(self.coords)
        # print("coord to handle: coord {}, coords {}".format(coord, self.coords))
        if len(self.coords) is 0 or coord > self.coords[-1]: 
            return None
        lo = 0
        hi = len(self.coords) - 1
        mid = 0
        while lo <= hi:
            mid = math.ceil((hi + lo) / 2)
            # print("target {}, lo: {}, hi: {}, mid {}, coord {}".format(coord, lo, hi, mid, self.coords[mid]))
            if self.coords[mid] == coord:
                # print()
                return mid
            elif self.coords[mid] < coord:
                lo = mid + 1
            else: # self.coords[mid] > coord:
                hi = mid - 1
        # print()
        if (coord > self.coords[mid]):
            mid += 1
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
            self.payloads = self.payloads + [None]
            return len(self.coords) - 1


        # if adding a new coord, make room for it
        if self.coords[handle_to_add] is not coord:
            # add coord to coord list
            self.coords = self.coords[:handle_to_add] + [coord] + self.coords[handle_to_add:]

            # move payloads to make space
            self.payloads = self.payloads[:handle_to_add] + [None] + self.payloads[handle_to_add:]

        return handle_to_add

    # return handle for termination
    def updatePayload(self, handle, payload):
        if handle is None:
            return None
        
        if handle >= 0 and handle < len(self.payloads):
            # print(self.payloads)
            # print("setting payload at {} to {}".format(handle, payload))
            self.payloads[handle] = payload
            # print(self.payloads)
        return handle

    # print this fiber representation in C
    def printFiber(self):
        print("coords: {}, payloads: {}".format(self.coords, self.payloads))
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
