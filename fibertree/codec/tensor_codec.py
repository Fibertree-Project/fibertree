import yaml
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor

#import compression groupings
from .compression_types import descriptor_to_fmt

# import compression formats
from .formats.uncompressed import Uncompressed
from .formats.coord_list import CoordinateList
from .formats.bitvector import Bitvector
from .formats.hashtable import HashTable

class Codec:
    # format descriptor should be a tuple of valid formats
    # order descriptor specified SoA or AoS at each rank (currently unused)
    # AoS / SoA doesn't apply to some formats (e.g. U) -> C (SoA, default should be here) / Ca (AoS)
    def __init__(self, format_descriptor):
        # take a list of compression formats
        # TODO: check that they are valid
        self.format_descriptor = format_descriptor
        
        # convert descriptor to list of formats
        self.fmts = list()
        for fmt in self.format_descriptor:
            self.fmts.append(descriptor_to_fmt[fmt])
        
        # assumes pre-flattened for now
        self.num_ranks = len(format_descriptor)
                 
    def get_format_descriptor(self):
        return self.format_descriptor
                 
    def get_num_ranks(self):
        return self.num_ranks

    # encode
    def encode(self, depth, a, ranks, output):
        if depth >= len(ranks):
            return -1
        # keys are in the form payloads_{rank name}, coords_{rank name}
        # deal with the root separately
        # TODO: make this a function
        payloads_key = "payloads_{}".format(ranks[depth].lower())
        coords_key = "coords_{}".format(ranks[depth].lower())

        if depth == -1:           
            # recurse one level down without adding to output yet
            size = self.encode(depth + 1, a, ranks, output)

            if self.fmts[depth + 1].encodeUpperPayload():
                # output[payloads_key].extend(occ_list)
            # store at most one payload at the root (size of first rank)
                payloads_key = "payloads_root"
                output[payloads_key].append(size)
            return None

        # otherwise, we are in the fibertree
        fmt = self.fmts[depth]
        # self.format_descriptor[depth]
        dim_len = a.getShape()[0]

        # 
        fiber_occupancy, occ_list = fmt.encodeFiber(a, dim_len, self, depth, ranks, output)
        return fiber_occupancy
 
    # encode
    # static functions
    # rank output dict based on rank names
    @staticmethod
    def get_output_dict(rank_names, format_descriptor):
            output = dict()
            output["payloads_root"] = []

            for i in range(0, len(rank_names)):
                    name = rank_names[i]
                    coords_key = "coords_{}".format(name.lower())
                    payloads_key = "payloads_{}".format(name.lower())

                    output[coords_key] = []
                    output[payloads_key] = []  
                    if format_descriptor[i] == "Hf":
                        ptrs_key = "ptrs_{}".format(name.lower())
                        ht_key = "ht_{}".format(name.lower())

                        output[ptrs_key] = []
                        output[ht_key] = []
            return output

    # given a tensor, descriptor, and dict of tensor encoded in that format
    # print and write out yaml in that format
    # TODO: change the output file name (currently just writes it to [descriptor string].yaml)
    @staticmethod
    def write_yaml(tensor, descriptor, tensor_in_format):
            # header
            header = dict()
            header["name"] = "tensor-a" # TODO: take this as input later
            header["rank_ids"] = tensor.getRankIds()
            header["shapes"] = tensor.getShape()
            header["formats"] = descriptor
            rank_names = tensor.getRankIds()

            # print(tensor_in_format)
            # hierarchical yaml according to ranks
            scratchpads = dict()
            if len(tensor_in_format["payloads_root"]) > 0:
                    scratchpads["rank_0"] = { "payloads" : tensor_in_format["payloads_root"] }

            # write one rank at a time
            for i in range(0, len(rank_names)):
                    rank_name = rank_names[i].lower()
                    coords_key = "coords_{}".format(rank_name)
                    payloads_key = "payloads_{}".format(rank_name)
                    ptrs_key = "ptrs_{}".format(rank_name)
                    ht_key = "ht_{}".format(rank_name)
                    key = "rank_" + str(i+1)
                    rank_dict = dict()
                    
                    # only write if scratchpad is nonempty
                    if len(tensor_in_format[coords_key]) > 0:
                        rank_dict["coords"] = tensor_in_format[coords_key]
                    if len(tensor_in_format[payloads_key]) > 0:
                        if descriptor[i] == "U" and i < len(rank_names) - 1:
                            rank_dict["offsets"] = tensor_in_format[payloads_key]
                        elif descriptor[i] == "Hf" and i < len(rank_names) - 1:
                            rank_dict["offsets"] = tensor_in_format[payloads_key]
                        else:
                            rank_dict["payloads"] = tensor_in_format[payloads_key]
                    if descriptor[i] == "Hf":
                        rank_dict["ptrs"] = tensor_in_format[ptrs_key]
                        rank_dict["bin_heads"] = tensor_in_format[ht_key]
                    if len(rank_dict) > 0:
                        scratchpads[key] = rank_dict
                    
            header["scratchpads"] = scratchpads
                    
            data = dict()
            data["tensor"] = header

            print(yaml.dump(data, default_flow_style=None, sort_keys=False))

            # outfilename = ''.join(descriptor) + '.yaml'
            # with open(outfilename, "w") as f:
                # print(yaml.dump(data, default_flow_style=None, sort_keys=False))
                # yaml.dump(data, f)

