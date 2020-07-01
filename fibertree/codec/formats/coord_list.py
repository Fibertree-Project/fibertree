from .compression_format import CompressionFormat

class CoordinateList(CompressionFormat):
    def __init__(self):
        self.name = "C"

    @staticmethod
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        coords_key = "coords_{}".format(ranks[depth].lower())
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        occ_list = list()
        prev_nz = 0
        
        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output)

            # keep track of actual occupancy (nnz in this fiber)
            # fiber_occupancy = fiber_occupancy + 1

            cumulative_occupancy = cumulative_occupancy + child_occupancy
            occ_list.append(cumulative_occupancy)
            # store coordinate explicitly
            coords = encodeCoord(prev_nz, ind)
            output[coords_key].extend(coords)
            fiber_occupancy = fiber_occupancy + len(coords)

            prev_nz = ind + 1

        return fiber_occupancy, occ_list

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
