from .compression_format import CompressionFormat
import operator
class Uncompressed(CompressionFormat):
    def __init__(self):
        self.name = "U"

    @staticmethod
    def getName(self):
        return self.getName()

    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        payloads_key = "payloads_{}".format(ranks[depth].lower())
        # init vars
        fiber_occupancy = 0
        
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] is "Hf": 
        # (codec.format_descriptor[depth + 1] is "Hf": 
                # or codec.format_descriptor[depth + 1] is "T"):
            cumulative_occupancy = [0, 0]
        occ_list = list()
        prev_nz = 0
        occ_list.append(cumulative_occupancy)
        
        # iterate through all coords (nz or not)
        for i in range(0, dim_len):
            # internal levels
            if depth < len(ranks) - 1:
                child_occupancy = codec.encode(depth + 1, a.getPayload(i), ranks, output)

                # keep track of occupancy (cumulative requires ordering)
                # cumulative_occupancy = cumulative_occupancy + child_occupancy
                # print(child_occupancy)
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                    # cumulative_occupancy = map(operator.add, cumulative_occupancy, child_occupancy)
                occ_list.append(cumulative_occupancy)
            else: # leaf level
                if a.getPayload(i) == 0:
                    output[payloads_key].append(0)
                else:
                    output[payloads_key].append(a.getPayload(i).value)
        
        # store payload if necessary
        if depth < len(ranks) - 1 and codec.fmts[depth+1].encodeUpperPayload():
            output[payloads_key].extend(occ_list)

        # return 1 so if upper levels encode payloads, 
        return 1, occ_list

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
