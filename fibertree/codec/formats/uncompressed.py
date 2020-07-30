from .compression_format import CompressionFormat
import operator
class Uncompressed(CompressionFormat):
    def __init__(self):
        self.name = "U"
        CompressionFormat.__init__(self)

    # instantiate this fiber in the format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        
        # init vars
        fiber_occupancy = 0
        
        cumulative_occupancy = codec.get_start_occ(depth)

        occ_list = list()

        # keep track of shape during encoding
        self.shape = dim_len

        # iterate through all coords (nz or not)
        for i in range(0, dim_len):
            # internal levels
            if depth < len(ranks) - 1:
                child_occupancy = codec.encode(depth + 1, a.getPayload(i), ranks, output, output_tensor)

                # keep track of occupancy (cumulative requires ordering)
                # cumulative_occupancy = cumulative_occupancy + child_occupancy
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)

            else: # leaf level
                if a.getPayload(i) == 0:
                    output[payloads_key].append(0)
                    self.payloads.append(0)
                else:
                    output[payloads_key].append(a.getPayload(i).value)
                    self.payloads.append(a.getPayload(i).value)
        
        # store payload if necessary
        if depth < len(ranks) - 1 and codec.fmts[depth+1].encodeUpperPayload():
            output[payloads_key].extend(occ_list)
            self.payloads.extend(occ_list)

        # return 1 so if upper levels encode payloads, 
        return 1, occ_list

    def getPayloads(self):
        return self.payloads

    def printFiber(self):
        print(self.payloads)

    def handleToCoord(self, handle):
        if handle is None or handle >= self.shape:
            return None
        return handle
    # API methods
    
    # max number of elements in a slice is proportional to the shape
    def getSliceMaxLength(self):
        return self.shape

    def coordToHandle(self, coord):
        if coord < 0 or coord >= self.shape:
            return None
        return coord
    
    def insertElement(self, coord):
        assert coord < self.shape
        return coord

    def updatePayload(self, handle, payload):
        assert handle < self.shape
        self.payloads[handle] = payload

    ### static methods
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return []

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
        return [0] * num_to_pad

    # implicit coords
    @staticmethod
    def encodeCoords():
        return False

    # implicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return False
