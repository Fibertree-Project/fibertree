from fibertree import Codec

# take an HFA tensor, convert it to compressed representation in python
def encodeSwoopTensorInFormat(tensor, descriptor):
    codec = Codec(tuple(descriptor), [True]*len(descriptor))

    # get output dict based on rank names
    rank_names = tensor.getRankIds()
    # print("encode tensor: rank names {}, descriptor {}".format(rank_names, descriptor))
    # TODO: move output dict generation into codec
    output = codec.get_output_dict(rank_names)
    # print("output dict {}".format(output))
    output_tensor = []
    for i in range(0, len(descriptor)+1):
            output_tensor.append(list())

    # print("encode, output {}".format(output_tensor))
    codec.encode(-1, tensor.getRoot(), tensor.getRankIds(), output, output_tensor)

    # name the fibers in order from left to right per-rank
    rank_idx = 0
    rank_names = ["root"] + tensor.getRankIds()

    for rank in output_tensor:
        fiber_idx = 0
        for fiber in rank:
            fiber_name = "_".join([tensor.getName(), rank_names[rank_idx], str(fiber_idx)])
            fiber.setName(fiber_name)
            # fiber.printFiber()
            fiber_idx += 1
        rank_idx += 1
    return output_tensor

# tensor is a 2d linearized tensor (one list per rank)
# dump all stats into output dict
def dumpAllStatsFromTensor(tensor, output):
    for rank in tensor:
        for fiber in rank:
            # fiber.printFiber()
            fiber.dumpStats(output)
