from fibertree import Codec
from boltons.cacheutils import LRU
# take an HFA tensor, convert it to compressed representation in python
def encodeSwoopTensorInFormat(tensor, descriptor, tensor_shape=None, cache_size=32):
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
    codec.encode(-1, tensor.getRoot(), tensor.getRankIds(), output, output_tensor, shape=tensor_shape)

    # name the fibers in order from left to right per-rank
    rank_idx = 0
    rank_names = ["root"] + tensor.getRankIds()
    # tensor_cache = dict()

    tensor_cache = LRU(max_size = cache_size)
    for rank in output_tensor:
        fiber_idx = 0
        for fiber in rank:
            fiber_name = "_".join([tensor.getName(), rank_names[rank_idx], str(fiber_idx)])
            fiber.setName(fiber_name)
            # fiber.printFiber()
            fiber.cache = tensor_cache
            fiber_idx += 1
        rank_idx += 1
    return output_tensor

# tensor is a 2d linearized tensor (one list per rank)
# dump all stats into output dict
def dumpAllStatsFromTensor(tensor, output, cache_output, name):
    for rank in tensor:
        for fiber in rank:
            # 
            fiber.dumpStats(output)
            # print("{} cache {}".format(fiber.name, fiber.cache))
            # fiber.printFiber()
    cache_output[name + '_buffer_access'] = tensor[0][0].cache.hit_count
    cache_output[name + '_DRAM_access'] = tensor[0][0].cache.miss_count
    # print("\thit count {}, miss count {}, soft miss count {}".format(fiber.cache.hit_count, fiber.cache.miss_count, fiber.cache.soft_miss_count))
