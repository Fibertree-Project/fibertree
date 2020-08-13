from .compression_format import CompressionFormat
import math


class TwoHandle():
    def __init__(self, coords_handle = None, payloads_handle = None):
        self.coords_handle = coords_handle
        self.payloads_handle = payloads_handle

class Bitvector(CompressionFormat):
    # untruncated bitvector
    def __init__(self):
        CompressionFormat.__init__(self)
        self.occupancies = list()
        self.bits_per_word = 32
        self.iter_handle = TwoHandle()

        self.prev_coord_at_payload = None
        self.prev_coord_word_write = None
    # instantiate current fiber in B format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1:
            if codec.format_descriptor[depth + 1] is "Hf" or codec.format_descriptor[depth + 1] is "T":
                cumulative_occupancy = [0, 0]
        occ_list = list()
        occ_list.append(cumulative_occupancy)
        prev_nz = 0
        self.coords = [0]*dim_len
        for ind, (val) in a:
            if depth < len(ranks) - 1:
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
                # store coordinate explicitly
                self.payloads.append(fiber)
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)


                if codec.fmts[depth+1].encodeUpperPayload():
                    output[payloads_key].append(cumulative_occupancy)
                    self.occupancies.append(cumulative_occupancy)

            # TODO: make this store more than one bit per entry
            # set coord bit
            self.coords[ind] = 1

            fiber_occupancy += 1
            # fiber_occupancy = fiber_occupancy + len(coords)           
    	    # encode payload if necessary
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)
                self.payloads.append(val.value)
            prev_nz = ind + 1

        # pad end if necessary
        output[coords_key].extend(self.coords)
        # print("encode fiber: coords {}, payloads {}".format(self.coords, self.payloads))
        return fiber_occupancy

    def getWordStart(self, index):
        return math.floor(float(index) / self.bits_per_word) * self.bits_per_word

    # TODO: fix at 0
    def getWordEnd(self, index):
        return math.ceil(float(index) / self.bits_per_word) * self.bits_per_word
    
    # return [lower, upper) bits of prev word searched
    def prevWordSearched(self):
        if self.prevHandleAtCoordSearched is None:
            return None, None
        lower = self.getWordStart(self.prevHandleAtCoordSearched) 
        assert lower % self.bits_per_word == 0
        upper = lower + self.bits_per_word
        # print("handle: {}, lower {}, upper {}".format(self.prevHandleAtCoordSearched, lower, upper))
        return lower, upper

    # handle = coord_handle
    def countCoordsRead(self, handle):
        lower, upper = self.prevWordSearched()
        #print("\t{} :: countCoordsRead handle {}, lower {}, upper {}".format(self.name, handle, lower, upper))
        if lower is None or handle < lower or handle >= upper:
            self.stats[self.coords_read_key] += 1
            self.prevHandleAtCoordSearched = self.getWordStart(handle)

    # handle = coord_handle
    def countCoordsWrite(self, handle):
        if self.prev_coord_word_write is None or handle < self.prev_coord_word_write or handle >= self.prev_coord_word_write + self.bits_per_word:
            self.prev_coord_word_write = self.getWordStart(handle)
            self.stats[self.coords_write_key] += 1

    # given a handle (index into bit vector) return the coord 
    def handleToCoord(self, iter_handle):
        assert(isinstance(iter_handle, TwoHandle)) 
        handle = iter_handle.coords_handle

        # print("{} handleToCoord, coord handle {}".format(self.name, handle))
        if handle is None or handle >= len(self.coords):
            return None
        # if nothing is saved
        self.countCoordsRead(handle)
        return handle

    def handleToPayload(self, iter_handle):
        # save the previous coord that we looked up for cost measure
        self.prev_coord_at_payload = iter_handle.coords_handle
        return super().handleToPayload(iter_handle.payloads_handle)
    # does this need to return a payload handle?
    # NOTE: in bitvector, handles to coords and payloads are different
    def coordToHandle(self, coord):
        return coord

    def getSize(self):
        size = math.ceil(len(self.coords) / self.bits_per_word) + len(self.occupancies)
        if not isinstance(self.payloads[0], CompressionFormat):
            size += len(self.payloads)
        return size

    # insertElement makes space for coord, payload and returns handle
    def insertElement(self, coord):
         if coord is None:
             return TwoHandle(None, None)
         coord_handle_to_add = self.handleToCoord(TwoHandle(coord))
         # cached
         if coord == self.prev_coord_at_payload:
            return TwoHandle(coord_handle_to_add, self.prevPayloadHandle)
         self.countCoordsRead(coord_handle_to_add)
         self.countCoordsWrite(coord_handle_to_add)
         # either way, need to count left
         payload_to_add_handle = self.countLeft(coord_handle_to_add)        
         # if unset, make space for it
         if self.coords[coord_handle_to_add] is 0:
             self.payloads = self.payloads[:payload_to_add_handle] + [0] + self.payloads[payload_to_add_handle:]
             self.stats[self.payloads_write_key] += len(self.payloads) - payload_to_add_handle
             self.coords[coord_handle_to_add] = 1
         return TwoHandle(coord_handle_to_add, payload_to_add_handle)
             
    def updatePayload(self, handle, payload):
        payload_handle = handle.payloads_handle
        if payload_handle is None:
            return None
        if payload_handle >= 0 and payload_handle < len(self.payloads):
            # we are always calling update payload right after inserting it, so don't need to count it twice
            # self.stats[self.payloads_write_key] += 1
            self.payloads[payload_handle] = payload
        return payload_handle

    def countLeft(self, coords_handle):
        # count_left has cost = number of words to the left
        # print("count left from {}, cur {}, to add {}".format(coords_handle, self.stats[self.coords_read_key], 
# coords_handle / self.bits_per_word))
        self.countCoordsRead(coords_handle)
        lower, upper = self.prevWordSearched()
        # count left is already in the previously readd word, so it does not
        # countributing to the cost
        if not (lower <= coords_handle and coords_handle < upper):
            self.stats[self.coords_read_key] += math.ceil(coords_handle / self.bits_per_word)
        result = 0 # count left 1s 
        for i in range(0, coords_handle):
            result += self.coords[i]
        return result

    # setup coord and payload handle
    def setupSlice(self, base = 0, bound = None, max_num = None):
        super().setupSlice(base, bound, max_num)
        # start payloads handle
        self.iter_handle.coords_handle = self.coordToHandle(base)
        self.iter_handle.payloads_handle = self.countLeft(self.coords_handle)
        print("setup slice in B: coords handle {}, payloads handle {}".format(self.iter_handle.coords_handle, self.iter_handle.payloads_handle))
    
    # iterate through coords, finding next nonempty coord
    # then move payloads forward by 1 (compressed payloads)
    def nextInSlice(self):
        # print("{} nextInSlice (B format): coords handle {}, payloads handle {}".format(self.name,self.iter_handle.coords_handle, self.iter_handle.payloads_handle))
        # print("len coords {}, len payloads {}".format(len(self.coords), len(self.payloads)))
        if self.iter_handle.coords_handle >= len(self.coords) or self.iter_handle.payloads_handle >= len(self.payloads):
            return None
        if self.num_to_ret is not None and self.num_to_ret < self.num_ret_so_far:
            return None
        while self.iter_handle.coords_handle < len(self.coords) and self.coords[self.iter_handle.coords_handle] is not 1:
            # print("iter coords handle: {}".format(self.iter_handle.coords_handle))
            self.iter_handle.coords_handle += 1
        
        # if in range, return 
        if self.iter_handle.coords_handle < len(self.coords):
            to_ret = TwoHandle(self.iter_handle.coords_handle, self.iter_handle.payloads_handle)
            self.iter_handle.coords_handle +=1 
            self.iter_handle.payloads_handle += 1
            self.num_ret_so_far += 1
            # print("\tnextInSlice returning ({}, {})".format(to_ret.coords_handle, to_ret.payloads_handle))
            return to_ret
        else:
            return None

    def printFiber(self):
        print("{} :: bitvector: {}, occupancies: {}, payloads: {}".format(self.name, self.coords, self.occupancies, self.payloads))
    
    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads because payloads at this level are compressed
    @staticmethod
    def encodeUpperPayload():
        return True
