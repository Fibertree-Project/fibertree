from .compression_format import CompressionFormat

# 
class Bitvector(CompressionFormat):
    # untruncated bitvector
    def __init__(self):
        self.name = "B"

    @staticmethod
    def getName(self):
        return self.getName()


    @staticmethod
    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        coords_key = "coords_{}".format(ranks[depth].lower())
        payloads_key = "payloads_{}".format(ranks[depth].lower())
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
            # store coordinate explicitly
            
            coords = Bitvector.encodeCoord(prev_nz, ind)
            output[coords_key].extend(coords)
            fiber_occupancy = fiber_occupancy + len(coords)

	    # encode payload if necessary
            if depth == len(ranks) - 1:
                output[payloads_key].append(val.value)
            elif codec.fmts[depth+1].encodeUpperPayload():
                output[payloads_key].append(cumulative_occupancy)

            prev_nz = ind + 1

        # pad end if necessary
        end_zeroes = Bitvector.endCoords(dim_len - prev_nz)
        output[coords_key].extend(end_zeroes)
        return fiber_occupancy, occ_list


    @staticmethod
    def encodeCoord(prev_ind, ind):
        output = list()	
        for i in range(prev_ind, ind):
            output.append(0)
        output.append(1)
        return output

    # in a bitvector, payloads are compressed
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

    # at end of coords scratchpad, pad with zeroes
    @staticmethod
    def endCoords(num_to_pad):
        return [0]*num_to_pad

    # no padding for payload scratchpad
    @staticmethod
    def endPayloads(num_to_pad):
        return []

    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # implicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return False
