import yaml
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor

#import compression groupings
from .compression_types import descriptor_to_fmt

class Codec:
    # format descriptor should be a tuple of valid formats
    # order descriptor specified SoA or AoS at each rank (currently unused)
    # AoS / SoA doesn't apply to some formats (e.g. U) -> C (SoA, default should be here) / Ca (AoS)
    def __init__(self, format_descriptor, cumulative_payloads):
        # take a list of compression formats
        # TODO: check that they are valid
        self.format_descriptor = format_descriptor
        self.cumulative_payloads = cumulative_payloads
        # print(cumulative_payloads)
        assert len(cumulative_payloads) == len(format_descriptor)
        # convert descriptor to list of formats
        self.fmts = list()
        for fmt in self.format_descriptor:
            self.fmts.append(descriptor_to_fmt[fmt])
        
        # assumes pre-flattened for now
        self.num_ranks = len(format_descriptor)

    def add_payload(self, depth, output, cumulative, noncumulative):
        if self.cumulative_payloads[depth]:
            output.append(cumulative)
        else:
            output.append(noncumulative)

    def get_format_descriptor(self):
        return self.format_descriptor

    
    def get_start_occ(self, depth):
        return self.fmts[depth].startOccupancy()

    @staticmethod
    def get_num_ranks(self):
        return self.num_ranks

    # return a list of occupancies per-rank 
    @staticmethod
    def get_occupancies(self, depth, a, num_ranks, output):
        if depth >= num_ranks:
            return 
        count = 0
        for ind, (val) in a:
            self.get_occupancies(depth + 1, val, num_ranks, output)
            count = count + 1
        output[depth] = output[depth] + count

    # return coords_{rank}, payloads_{rank}
    @staticmethod
    def get_keys(ranks, depth):
        assert depth < len(ranks)
        return "coords_{}".format(ranks[depth].lower()), "payloads_{}".format(ranks[depth].lower()),

    # encode
    def encode(self, depth, a, ranks, output, output_tensor):
        if depth >= len(ranks):
            return -1
        # keys are in the form payloads_{rank name}, coords_{rank name}
        # deal with the root separately
        coords_key, payloads_key = self.get_keys(ranks, depth)

        if depth == -1:           
            # recurse one level down without adding to output yet
            size = self.encode(depth + 1, a, ranks, output, output_tensor)

            if self.fmts[depth + 1].encodeUpperPayload():
                # store at most one payload at the root (size of first rank)
                payloads_key = "payloads_root"
                output[payloads_key].append(size)
            return None

        # otherwise, we are in the fibertree
        fmt = self.fmts[depth]
        fiber = fmt()
        # fmt = self.fmts[depth]
        # self.format_descriptor[depth]
        dim_len = a.getShape()[0]

        fiber_occupancy, occ_list = fiber.encodeFiber(a, dim_len, self, depth, ranks, output, output_tensor)
        output_tensor[depth].append(fiber)
        return fiber_occupancy
 
    # encode
    # static functions
    # rank output dict based on rank names
    # @staticmethod
    # def get_output_dict(rank_names, format_descriptor):
    def get_output_dict(self, rank_names):
            output = dict()
            output["payloads_root"] = []

            # print("in output dict {}".format(self))
            for i in range(0, len(rank_names)):
                    coords_key, payloads_key = Codec.get_keys(rank_names, i)

                    output[coords_key] = []
                    output[payloads_key] = []  
                    if self.format_descriptor[i] == "Hf":
                        ptrs_key = "ptrs_{}".format(rank_names[i].lower())
                        ht_key = "ht_{}".format(rank_names[i].lower())

                        output[ptrs_key] = []
                        output[ht_key] = []
            return output

    # given a tensor, descriptor, and dict of tensor encoded in that format
    # print and write out yaml in that format
    # TODO: change the output file name (currently just writes it to [descriptor string].yaml)
    # @staticmethod
    def write_yaml(self, tensor, rank_names, descriptor, tensor_in_format):
            # header
            header = dict()
            header["name"] = "tensor-a" # TODO: take this as input later
            header["rank_ids"] = tensor.getRankIds()
            header["shapes"] = tensor.getShape()
            header["formats"] = descriptor
            occupancies = [0]*len(rank_names)
            self.get_occupancies(0, tensor.getRoot(), len(rank_names), occupancies)

            header["occupancies"] = occupancies
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

