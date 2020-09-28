import yaml
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor
from .formats.uncompressed import Uncompressed
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
    def encode(self, depth, a, ranks, output, output_tensor, shape=None):
        if depth >= len(ranks):
            return -1
        # keys are in the form payloads_{rank name}, coords_{rank name}
        # deal with the root separately
        coords_key, payloads_key = self.get_keys(ranks, depth)

        if depth == -1:           
            # recurse one level down without adding to output yet
            root, size = self.encode(depth + 1, a, ranks, output, output_tensor, shape=shape)
            HFA_root = Uncompressed()
            HFA_root.shape = 1
            # print("HFA root, next fmt {}".format(self.fmts[0]))
            if self.fmts[0].encodeUpperPayload():
                # store at most one payload at the root (size of first rank)
                payloads_key = "payloads_root"
                output[payloads_key].append(size)
                HFA_root.occupancies = [0]
                HFA_root.count_payload_reads = True

            HFA_root.payloads = [root]
            output_tensor[0] = [HFA_root]
            return root, size

        # otherwise, we are in the fibertree
        fmt = self.fmts[depth]
        fiber = fmt()
        dim_len = a.getShape()[0]
        # print("shape arg {}".format(shape))
        if shape != None:
            dim_len = shape[depth]
            # print("depth {}, HFA shape {}, real shape {}".format(depth, a.getShape()[0], dim_len))
            assert dim_len >= a.getShape()[0]
        stats_key = ranks[depth] + "_" + str(len(output_tensor[depth+1]))
        fiber.setName(stats_key)
        fiber_occupancy = fiber.encodeFiber(a, dim_len, self, depth, ranks, output, output_tensor, shape=shape)
        fiber.nnz = fiber_occupancy
        fiber.idx_in_rank = len(output_tensor[depth + 1])
        if len(output_tensor[depth + 1]) == 0:
            fiber.occupancy_so_far = 0
        else:
            # exclusive prefix for indexing
            if isinstance(fiber_occupancy, int):
                # fiber.occupancy_so_far = fiber_occupancy + output_tensor[depth + 1][-1].occupancy_so_far
                fiber.occupancy_so_far = output_tensor[depth + 1][-1].occupancy_so_far + output_tensor[depth + 1][-1].nnz
                # print("in encode, name {}, occupancy so far {}".format(fiber.name, fiber.occupancy_so_far))
            else:
                print(fiber_occupancy)
                assert isinstance(fiber_occupancy, list)
                fiber.occupancy_so_far = fiber_occupancy[0] + output_tensor[depth + 1][-1].occupancy_so_far

        output_tensor[depth+1].append(fiber)
        
        # print("\tencode at depth {}: {}".format(depth+1, output_tensor[depth+1]))
        return fiber, fiber_occupancy
 
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
                    if self.format_descriptor[i] == "H":
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
                        elif descriptor[i] == "H" and i < len(rank_names) - 1:
                            rank_dict["offsets"] = tensor_in_format[payloads_key]
                        else:
                            rank_dict["payloads"] = tensor_in_format[payloads_key]
                    if descriptor[i] == "H":
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

