import yaml
from fibertree import Payload
from fibertree import Fiber
from fibertree import Tensor

#import compression groupings
from .compression_types import descriptor_to_fmt

# import compression formats
from .formats.uncompressed import Uncompressed
from .formats.coord_list import CoordinateList

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
        # keys are in the form payloads_{rank name}, coords_{rank name}
        payloads_key = "payloads_{}".format(ranks[depth].lower())
        coords_key = "coords_{}".format(ranks[depth].lower())

        # deal with the root separately
        if depth == -1:           
            # recurse one level down without adding to output yet
            size = self.encode(depth + 1, a, ranks, output)

            if self.fmts[depth + 1].encodeUpperPayload():
                # output[payloads_key].extend(occ_list)
            # store at most one payload at the root (size of first rank)
                payloads_key = "payloads_root"
                output[payloads_key].append(size)
            return None
        
        fmt = self.fmts[depth]
        # self.format_descriptor[depth]
        dim_len = a.getShape()[0]
        
        # leaf level
        if depth == self.num_ranks - 1:
            # keep track of the occupancy of this fiber 
            occupancy = 0
            
            # if U, may have to add some zeroes, so we need indexing
            prev_payloads_nz = 0
            prev_coords_nz = 0
            
            # iterate nonzeroes in the fiber
            for ind, (val) in a:
                assert isinstance(val, Payload)
                
                # if coords are implicit, add zeroes between nzs
                # TODO: make a list of the format classes and call from those
                to_add = self.fmts[depth].encodePayload(prev_payloads_nz, ind, val.value)
                prev_payloads_nz = ind + 1
                output[payloads_key].extend(to_add)

                # encode coords
                # if this rank has explicit coords
                coords = self.fmts[depth].encodeCoord(prev_coords_nz, ind)
                occupancy = occupancy + len(coords)
                prev_coords_nz = ind
                output[coords_key].extend(coords)

            output[payloads_key].extend(self.fmts[depth].endPayloads(dim_len - prev_payloads_nz))
            # if coords are implicit, fill in zeroes at end of payloads
            """
            if self.format_descriptor[depth] in implicit_coords:
                for i in range(prev_payloads_nz, dim_len):
                    output[payloads_key].append(0)
            """     
            return occupancy
                
        # internal levels
        else:
            # TODO: map labels to types
            next_fmt = self.fmts[depth + 1] 
           
            # keep track of occupancy of children and at current height
            cumulative_occupancy = 0
            fiber_occupancy = 0
            prev_nz = 0

            # TODO: can you merge the iterations? one is over nz, while the other is over dim_len
            # if coords at this depth are implicit, recurse on *every* coordinate (may be empty)
            occ_list = list()
            if not fmt.encodeCoords():
                for i in range(0, dim_len):
                    child_occupancy = self.encode(depth + 1, a.getPayload(i), ranks, output)
                    
                    # keep track of actual occupancy
                    if not a.getPayload(i).isEmpty():
                        fiber_occupancy = fiber_occupancy + 1
                    
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                    occ_list.append(cumulative_occupancy)
                        
            # if coords at this depth are explicit, only the nonzeroes appear
            # at lower ranks             
            else:
                # iterate through nonzeroes at this rank
                for ind, (val) in a:
                    assert isinstance(val, Fiber)

                    # recursive call to sub-fibers (DFS traversal)
                    child_occupancy = self.encode(depth + 1, val, ranks, output)

                    # keep track of cumulative occupancy
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                    occ_list.append(cumulative_occupancy)

                    # store coordinate explicitly
                    coords = CoordinateList.encodeCoord(prev_nz, ind)
                    output[coords_key].extend(coords)
                    fiber_occupancy = fiber_occupancy + len(coords)
                    
                    prev_nz = ind + 1

            # whether there are payloads here depends on the format of the next rank
            # store occupancy in previous payloads if necessary
            if next_fmt.encodeUpperPayload():
                output[payloads_key].extend(occ_list)
            
            return fiber_occupancy
 
    # encode
    # static functions
    # rank output dict based on rank names
    @staticmethod
    def get_output_dict(rank_names):
            output = dict()
            output["payloads_root"] = []

            for name in rank_names:
                    coords_key = "coords_{}".format(name.lower())
                    payloads_key = "payloads_{}".format(name.lower())

                    output[coords_key] = []
                    output[payloads_key] = []  
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

            # hierarchical yaml according to ranks
            scratchpads = dict()
            if len(tensor_in_format["payloads_root"]) > 0:
                    scratchpads["root"] = { "payloads" : tensor_in_format["payloads_root"] }
            
            # write one rank at a time
            for i in range(0, len(rank_names)):
                    rank_name = rank_names[i].lower()
                    coords_key = "coords_{}".format(rank_name)
                    payloads_key = "payloads_{}".format(rank_name)
                    key = "rank_" + str(i)
                    rank_dict = dict()
                    
                    # only write if scratchpad is nonempty
                    if len(tensor_in_format[coords_key]) > 0:
                        rank_dict["coords"] = tensor_in_format[coords_key]
                    if len(tensor_in_format[payloads_key]) > 0:
                        rank_dict["payloads"] = tensor_in_format[payloads_key]
                    
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

