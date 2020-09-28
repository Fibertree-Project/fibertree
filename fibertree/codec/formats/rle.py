from .compression_format import CompressionFormat

class RunLengthEncoding(CompressionFormat):
    def __init__(self):
        self.name = "R"

    @staticmethod
    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        
        # init vars
        fiber_occupancy = 0

        # TODO: HT to one payload
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] == "Hf":
    	    cumulative_occupancy = [0, 0] 

        occ_list = list()
        # occ_list.append(cumulative_occupancy)
        prev_nz = 0
        
        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output)
            
            if isinstance(cumulative_occupancy, int):
                cumulative_occupancy = cumulative_occupancy + child_occupancy
            else:
                cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
            
            # add cumulative or non-cumulative depending on settings
            codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)
            
            # store coordinate explicitly
            coords = RunLengthEncoding.encodeCoord(prev_nz, ind)
            output[coords_key].extend(coords)

            # keep track of nnz in this fiber
            fiber_occupancy = fiber_occupancy + 1

            # if at leaves, store payloads directly
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)

            prev_nz = ind
        
        # explicit payloads for next level
        if depth < len(ranks) - 1 and codec.fmts[depth+1].encodeUpperPayload():
            output[payloads_key].extend(occ_list)
        return fiber_occupancy, occ_list

    # encode coord explicitly
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return [ind - prev_ind]

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
