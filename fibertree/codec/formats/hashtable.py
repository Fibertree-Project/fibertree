from .compression_format import CompressionFormat
import operator
# hash table per fiber

class HashTable(CompressionFormat):
    def __init__(self):
        self.name = "Hf"

    @staticmethod
    def getName(self):
        return self.getName()

    @staticmethod
    def helper_add(output, key, to_add):
        if key in output:
            output[key].extend(to_add)
        else:
            output[key] = to_add

    @staticmethod
    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1 and codec.format_descriptor[depth + 1] is "Hf":
            cumulative_occupancy = (0, 0)
        occ_list = list()
        num_coords = len(a.getCoords())

        # if the hashtable length is fixed, don't need to write it as a payload
        hashtable_len = 6

        # init scratchpads
        ht = [None] * hashtable_len
        ptrs = list()
        coords = list()
        payloads = list()

        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output)
            # print(child_occupancy)
            # print(cumulative_occupancy)
            # TODO: make this a function
            # cumulative_occupancy = cumulative_occupancy + child_occupancy
            if isinstance(cumulative_occupancy, int):
                cumulative_occupancy = cumulative_occupancy + child_occupancy
            else:
                cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
            occ_list.append(cumulative_occupancy)
            # encode coord
            hash_key = ind % hashtable_len
            # print("ind: {}, hash key {}".format(ind, hash_key))
            bin_head = ht[hash_key]
            while bin_head is not None:
                # print("\tbin head {}".format(bin_head))
                if coords[bin_head] is ind:
                    # update payload or return because found
                    payloads[bin_head] = (val)
                    return True
                bin_head = ptrs[bin_head]

            ptrs.append(ht[hash_key])
            ht[hash_key] = len(ptrs) - 1 
            coords.append(ind)
            assert(len(ptrs) == len(coords))

            # add to payloads
            # if at the leaves, add the actual payloads
            if depth == len(ranks) - 1:
                payloads.append(val.value)
            elif codec.fmts[depth + 1].encodeUpperPayload():
                payloads.append(cumulative_occupancy)
            else:
                payloads.append(fiber_occupancy)
            
            fiber_occupancy = fiber_occupancy + 1

        coords_key, payloads_key = codec.get_keys(ranks, depth)
        ptrs_key = "ptrs_{}".format(ranks[depth].lower())
        ht_key = "ht_{}".format(ranks[depth].lower())

        output[coords_key].extend(coords)
        output[payloads_key].extend(payloads)
        output[ptrs_key].extend(ptrs)
        output[ht_key].extend(ht)

        # linearize output dict
        # coords in the format of two lists: 
        # 1. like the segment table in CSR, that points to the start of what was in that bucket
        # 2. linearization of buckets in contiguous order

        """
        print("ht " + str(ht))
        print("ptrs " + str(ptrs))
        print("coords " + str(coords))
        print("payloads " + str(payloads))
        """
        total_size = hashtable_len + len(ptrs) + len(coords) + len(payloads)
        return [fiber_occupancy, hashtable_len], occ_list

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
        return []

    # implicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return True
    
    @staticmethod 
    def startOccupancy():
        return [0, 0]
