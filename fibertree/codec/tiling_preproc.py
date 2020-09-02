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

def preproc_mtx_dsds():
    tensor_name = sys.argv[1]
    infilename = sys.argv[2]
    outfilename = sys.argv[3]
    outdir = sys.argv[4]
    splits = sys.argv[5].split(',') # get tilings

    # matrix market to the YAML that HFA reads
    mm_to_hfa_yaml(infilename, tensor_name, ['S', 'D'], outdir + outfilename)
    t0 = time.clock()
    # test reading the yaml into HFA
    a_sd = Tensor.fromYAMLfile(outdir + outfilename)
    t1 = time.clock() - t0
    print("time to read into HFA: {}".format(t1)) # cpu seconds
    a_sd.dump(outdir +"sd_" + outfilename)
    
    # swap (S, D) to (D, S)
    t0 = time.clock()
    a_ds = a_sd.swapRanks()
    t1 = time.clock() - t0
    print("time to swap S, D in HFA: {}".format(t1)) # cpu seconds
    a_ds.dump(outdir +"ds_" + outfilename)

    # split D
    t0 = time.clock()
    a_dds = a_ds.splitUniform(int(splits[0])) # split D
    t1 = time.clock() - t0
    print("time to splitUniform DS on D {}".format(t1)) # cpu seconds
    a_dds.dump(outdir +"dds_" + outfilename)

    # split S
    t0 = time.clock()
    a_ddss = a_dds.splitUniform(int(splits[1]), depth=2)
    t1 = time.clock() - t0
    print("time to splitUniform DS on S: {}".format(t1)) # cpu seconds 
    a_ddss.dump(outdir +"ddss_" + outfilename)

    # DDSS -> DSDS
    t0 = time.clock()
    a_dsds = a_ddss.swapRanks(depth=1)
    t1 = time.clock() - t0
    print("time to swap intermediate D, S in HFA: {}".format(t1)) # cpu seconds
    a_dsds.dump(outdir +"dsds_" + outfilename)

def preproc_mtx_sdsd():
    tensor_name = sys.argv[1]
    
    # input in matrix market
    infilename = sys.argv[2]
    
    # output file suffix (.yaml)
    outfilename = sys.argv[3]
    outdir = sys.argv[4]

    splits = sys.argv[5].split(',') # get tilings
    
    # matrix market to the YAML that HFA reads
    mm_to_hfa_yaml(infilename, tensor_name, ['S', 'D'], outdir + outfilename)
    t0 = time.clock()
    # test reading the yaml into HFA
    a_sd = Tensor.fromYAMLfile(outdir + outfilename)
    t1 = time.clock() - t0
    print("time to read into HFA: {}".format(t1)) # cpu seconds
    a_sd.dump(outdir +"sd_" + outfilename)
 
    # split S
    t0 = time.clock()
    a_ssd = a_sd.splitUniform(int(splits[0]), relativeCoords=False)
    t1 = time.clock() - t0
    print("time to splitUniform SD on S: {}".format(t1)) # cpu seconds 
    a_ssd.dump(outdir +"ssd_" + outfilename)
    
    # split D
    t0 = time.clock()
    a_ssdd = a_ssd.splitUniform(int(splits[1]), depth=2, relativeCoords=False) # split D
    t1 = time.clock() - t0
    print("time to splitUniform SSD on D {}".format(t1)) # cpu seconds
    a_ssdd.dump(outdir +"ssdd_" + outfilename)

    # SSDD -> SDSD
    t0 = time.clock()
    a_sdsd = a_ssdd.swapRanks(depth=1)
    t1 = time.clock() - t0
    print("time to swap intermediate D, S in HFA: {}".format(t1)) # cpu seconds
    a_sdsd.dump(outdir +"sdsd_" + outfilename)

if __name__ == "__main__":
    preproc_mtx_sdsd()
    preproc_mtx_dsds()
