import yaml
from fibertree import Tensor
import sys
import time
from fibertree import Codec

# matrix market converter
def mm_to_hfa_yaml(infilename, tensor_name, rank_ids, outfilename):
    # read mm into csr
    shape_m = 0
    shape_n = 0
    nnz = 0
    csr = None
    m_coords = list()
    with open(infilename, 'r') as infile:
        for line in infile:
            if line.startswith("%"):
                continue
            parts = line.split(" ")
            elt_1 = int(parts[0])
            elt_2 = int(parts[1])
            elt_3 = int(parts[2])
            if shape_m == 0: # not filled in yet
                shape_m = elt_1
                shape_n = elt_2
                nnz = elt_3
                csr = list()
                for i in range(0, shape_m):
                    csr.append([list(), list()])
            else:
                # 0 indexing
                elt_1 -= 1
                elt_2 -= 1
                
                assert elt_1 < shape_m
                # struct of arrays for (coord, payload)
                if len(m_coords) == 0 or m_coords[-1] != elt_1:
                    m_coords.append(elt_1)
               
                csr[elt_1][0].append(elt_2)
                csr[elt_1][1].append(elt_3)

    print("finished reading in CSR")
    # write to YAML (manually? since fiber is not a unique key)
    twospace = "  "
    fourspace = twospace * 2
    sixspace = fourspace + twospace
    eightspace = fourspace * 2
    tenspace = eightspace + twospace
    twelvespace = sixspace * 2
    fourteenspace = twelvespace + twospace
    with open(outfilename, 'w') as outfile:
        outfile.write("tensor:\n")
        outfile.write(twospace + "name: {}\n".format(tensor_name))
        outfile.write(twospace + "rank_ids: {}\n".format(rank_ids))
        outfile.write(twospace + "shape: {}\n".format([shape_m, shape_n]))
        outfile.write(twospace + "root:\n")
        outfile.write(fourspace + "- fiber:\n")
        outfile.write(eightspace + "coords: {}\n".format(m_coords))
        outfile.write(eightspace + "payloads:\n")
        for i in range(0, len(m_coords)):
            outfile.write(tenspace + "- fiber:\n")
            outfile.write(fourteenspace + "coords: {}\n".format(csr[m_coords[i]][0]))
            outfile.write(fourteenspace + "payloads: {}\n".format(csr[m_coords[i]][1]))
    print("finished writing out YAML")

def preproc_mtx():
    tensor_name = sys.argv[1]
    infilename = sys.argv[2]
    outfilename = sys.argv[3]

    # matrix market to the YAML that HFA reads
    mm_to_hfa_yaml(infilename, tensor_name, ['S', 'D'], outfilename)
    t0 = time.clock()
    # test reading the yaml into HFA
    a_sd = Tensor.fromYAMLfile(outfilename)
    t1 = time.clock() - t0
    print("time to read into HFA: {}".format(t1)) # cpu seconds
    a_sd.dump("sd_" + outfilename)
    
    print("occupancy of S rank {}".format(len(a_sd.getRoot().coords)))
    # swap (S, D) to (D, S)
    t0 = time.clock()
    a_ds = a_sd.swapRanks()
    # a_ds.shape = [a_sd.shape[1], a_sd.shape[0]]
    print("occupancy of D rank {}".format(len(a_ds.getRoot().coords)))
    t1 = time.clock() - t0
    print("time to swap S, D in HFA: {}".format(t1)) # cpu seconds
    a_ds.dump("ds_" + outfilename)

    # split D
    t0 = time.clock()
    a_ds_split_uniform = a_ds.splitUniform(256, relativeCoords=False) # split D
    t1 = time.clock() - t0
    print("time to splitUniform DS on D {}".format(t1)) # cpu seconds
    a_ds_split_uniform.dump("dds_" + outfilename)


    # split S
    t0 = time.clock()
    a_ddss = a_ds_split_uniform.splitUniform(32, depth=2, relativeCoords=False)
    t1 = time.clock() - t0
    print("time to splitUniform DS on S: {}".format(t1)) # cpu seconds 
    a_ddss.dump("ddss_" + outfilename)

    # DDSS -> DSDS
    t0 = time.clock()
    a_dsds = a_ddss.swapRanks(depth=1)
    t1 = time.clock() - t0
    print("time to swap intermediate D, S in HFA: {}".format(t1)) # cpu seconds
    a_dsds.dump("dsds_" + outfilename)


def preproc_mtx_sdsd():
    tensor_name = sys.argv[1]
    infilename = sys.argv[2]
    outfilename = sys.argv[3]

    # matrix market to the YAML that HFA reads
    mm_to_hfa_yaml(infilename, tensor_name, ['S', 'D'], outfilename)
    t0 = time.clock()
    # test reading the yaml into HFA
    a_sd = Tensor.fromYAMLfile(outfilename)
    t1 = time.clock() - t0
    print("time to read into HFA: {}".format(t1)) # cpu seconds
    a_sd.dump("sd_" + outfilename)
 
    # split S
    t0 = time.clock()
    a_ssd = a_sd.splitUniform(256, relativeCoords=False)
    t1 = time.clock() - t0
    print("time to splitUniform SD on S: {}".format(t1)) # cpu seconds 
    a_ssd.dump("ssd_" + outfilename)
    
    # split D
    t0 = time.clock()
    a_ssdd = a_ssd.splitUniform(32, depth=2, relativeCoords=False) # split D
    t1 = time.clock() - t0
    print("time to splitUniform SSD on D {}".format(t1)) # cpu seconds
    a_ssdd.dump("ssdd_" + outfilename)

    # SSDD -> SDSD
    t0 = time.clock()
    a_sdsd = a_ssdd.swapRanks(depth=1)
    t1 = time.clock() - t0
    print("time to swap intermediate D, S in HFA: {}".format(t1)) # cpu seconds
    a_sdsd.dump("sdsd_" + outfilename)

def encodeTensorInFormat(tensor, descriptor):
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

if __name__ == "__main__":
    preproc_mtx_sdsd()
    """
    t0 = time.clock()
    a_dsds = Tensor.fromYAMLfile(sys.argv[1])
    t1 = time.clock() - t0
    print("time to read DSDS from YAML into HFA: {}".format(t1)) # cpu seconds
 
    t0 = time.clock()
    tensor = encodeTensorInFormat(a_dsds, ["C", "C", "C", "C"])
    t1 = time.clock() - t0
    print("time encode HFA with codec: {}".format(t1)) # cpu seconds 
    """
