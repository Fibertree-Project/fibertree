from .compression_format import CompressionFormat

class CoordinateList(CompressionFormat):
    def __init__(self):
        self.name = "C"

    @staticmethod
    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        coords_key = "coords_{}".format(ranks[depth].lower())
        payloads_key = "payloads_{}".format(ranks[depth].lower())
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and (codec.format_descriptor[depth + 1] is "Hf" or
                codec.format_descriptor[depth+1] is "T"):
    	    cumulative_occupancy = [0, 0] 
        occ_list = list()
        occ_list.append(cumulative_occupancy)
        prev_nz = 0
        
        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output)
            # keep track of actual occupancy (nnz in this fiber)
            
            # print("ind {}, depth{}, child {}, cumulative {}".format(ind, depth, child_occupancy, cumulative_occupancy))
            if isinstance(cumulative_occupancy, int):
                cumulative_occupancy = cumulative_occupancy + child_occupancy
            else:
                cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
            occ_list.append(cumulative_occupancy)
            # store coordinate explicitly
            coords = CoordinateList.encodeCoord(prev_nz, ind)
            output[coords_key].extend(coords)
            # fiber_occupancy = fiber_occupancy + len(coords)
            fiber_occupancy = fiber_occupancy + 1
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)

            prev_nz = ind + 1
        
        # explicit payloads for next level
        if depth < len(ranks) - 1 and codec.fmts[depth+1].encodeUpperPayload():
            output[payloads_key].extend(occ_list)
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
