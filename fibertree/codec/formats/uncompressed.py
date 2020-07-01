from .compression_format import CompressionFormat

class Uncompressed(CompressionFormat):
    def __init__(self):
        self.name = "U"

    @staticmethod
    def getName(self):
        return self.getName()

    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        occ_list = list()
        
        for i in range(0, dim_len):
            child_occupancy = codec.encode(depth + 1, a.getPayload(i), ranks, output)

            # keep track of actual occupancy (nnz in this fiber)
            if not a.getPayload(i).isEmpty():
                fiber_occupancy = fiber_occupancy + 1
            cumulative_occupancy = cumulative_occupancy + child_occupancy
            occ_list.append(cumulative_occupancy)
        return fiber_occupancy, occ_list

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
