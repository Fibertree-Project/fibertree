from swoop import *
from swoop_util import *
from fibertree import Tensor
import sys
import yaml
import time

def verifyInHFA(A_HFA, B_HFA, Z_HFA):
    # HFA for verification
    b_n1 = B_HFA.getRoot()
    a_k1 = A_HFA.getRoot()
    z_n1 = Z_HFA.getRoot()
    for n1, (z_n0, b_k1) in z_n1 << b_n1:
      for k1, (a_k0, b_n0) in a_k1 & b_k1:
        for n0, (z, b_k0) in z_n0 << b_n0:
          for k0, (a, b) in a_k0 & b_k0:
            z += a * b


## Test program: Tiled Z-Stationary vector-matrix multiplication
#
#   Z_n = A_k * B_kn
# Tiled:
#   Z_n1n0 = A_k1k0 * B_k1n1k0n0
#
#for n1, (z_n1, b_k1) in z_n1 << b_n1:
#  for k1, (a_k0, b_k0) in a_k1 & b_k1:
#    for n0, (z, b_k0) in z_n0 << b_n0:
#      for k0, (a, b) in a_k0 & b_k0:
#        z += a * b

a = SwoopTensor(name="A", rank_ids=["K1", "K0"])
b = SwoopTensor(name="B", rank_ids=["N1", "K1", "N0", "K0"])
z = SwoopTensor(name="Z", rank_ids=["N1", "N0"])

a_k1 = a.getStartHandle()
b_n1 = b.getStartHandle()
z_root = z.getRootHandle()
z_n1 = z.getStartHandle()

# z_n1 << b_n1
b_n1_handles = Scan(b_n1)
b_n1_coords = HandlesToCoords(b_n1, b_n1_handles)
b_n1_payloads = HandlesToPayloads(b_n1, b_n1_handles)
(z_n1_handles, z_n1_new_fiber_handle) = InsertionScan(z_n1, b_n1_coords)
z_n1_payloads = HandlesToPayloads(z_n1, z_n1_handles)
b_k1s = PayloadsToFiberHandles(b_n1, b_n1_payloads)
z_n0s = PayloadsToFiberHandles(z_n1, z_n1_payloads)


# a_k1 & b_k1
b_k1_handless = Scan(b_k1s)
# Repeat a_k1 iteration for each b_k1
a_k1s = Amplify(a_k1, b_k1s, instance_name="K1")
a_k1_handless = Scan(a_k1s) 
a_k1_coordss = HandlesToCoords(a_k1s, a_k1_handless)
b_k1_coordss = HandlesToCoords(b_k1s, b_k1_handless)
(ab_k1_coordss, ab_a_k1_handless, ab_b_k1_handless) = Intersect(a_k1_coordss, a_k1_handless, b_k1_coordss, b_k1_handless, instance_name="K1")
ab_a_k1_payloadss = HandlesToPayloads(a_k1s, ab_a_k1_handless)
ab_b_k1_payloadss = HandlesToPayloads(b_k1s, ab_b_k1_handless)
a_k0ss = PayloadsToFiberHandles(a_k1s, ab_a_k1_payloadss)
b_n0ss = PayloadsToFiberHandles(b_k1s, ab_b_k1_payloadss)

# z_n0 << b_n0
b_n0_handlesss = Scan(b_n0ss)
b_n0_coordsss = HandlesToCoords(b_n0ss, b_n0_handlesss)
b_n0_payloadsss = HandlesToPayloads(b_n0ss, b_n0_handlesss)
# Repeat z_n0 iteration for each b_n0
z_n0ss = Amplify(z_n0s, b_n0ss, instance_name="N0")
(z_n0_handlesss, z_n0_new_fiber_handless) = InsertionScan(z_n0ss, b_n0_coordsss)
z_n0_payloadsss = HandlesToPayloads(z_n0ss, z_n0_handlesss)
b_k0sss = PayloadsToFiberHandles(b_n0ss, b_n0_payloadsss)
z_valuesss = PayloadsToValues(z_n0ss, z_n0_payloadsss)

# a_k0 & b_k0
b_k0_handlessss = Scan(b_k0sss)
# Repeat a_k0 iteration for each b_k0
a_k0sss = Amplify(a_k0ss, b_k0sss, instance_name="K0")
a_k0_handlessss = Scan(a_k0sss)
a_k0_coordssss = HandlesToCoords(a_k0sss, a_k0_handlessss)
b_k0_coordssss = HandlesToCoords(b_k0sss, b_k0_handlessss)
(ab_k0_coordssss, ab_a_k0_handlessss, ab_b_k0_handlessss) = Intersect(a_k0_coordssss, a_k0_handlessss, b_k0_coordssss, b_k0_handlessss, instance_name="K0")
ab_a_k0_payloadssss = HandlesToPayloads(a_k0sss, ab_a_k0_handlessss)
ab_b_k0_payloadssss = HandlesToPayloads(b_k0sss, ab_b_k0_handlessss)
a_valuessss = PayloadsToValues(a_k0sss, ab_a_k0_payloadssss)
b_valuessss = PayloadsToValues(b_k0sss, ab_b_k0_payloadssss)


# z_ref += a_val * b_val
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val: a_val * b_val
partial_productssss = Compute(body_func, a_valuessss, b_valuessss)
# Reduce into the same value until end of rank
z_new_valuesss = Reduce(partial_productssss, z_valuesss, instance_name="K0")
#z_new_valuesss = Reduce(partial_productssss, instance_name="K0")
z_n0_update_acksss = UpdatePayloads(z_n0ss, z_n0_handlesss, z_new_valuesss)

# Update N0 occupancy. (Should we be reducing here somehow?)
z_n1_handless = Amplify(z_n1_handles, b_n0ss, instance_name="N1_Upd")
z_n1s = Amplify(z_n1, ab_k1_coordss)
z_n1_update_acks = UpdatePayloads(z_n1s, z_n1_handless, z_n0_new_fiber_handless)

# Update root occupancy
z_root_update_ack = UpdatePayloads(z_root, Stream0(0), z_n1_new_fiber_handle)

# get the descriptor
str_desc = sys.argv[1]
frontier_descriptor = [str_desc[0], str_desc[1]]

"""
# read in inputs
# jhu_len = 5157
shape = 500 # TODO: take this as input
# generate input frontier and tile it
A_data = [0] * shape

# read in frontier
count = 1
a_file = sys.argv[2]
A_HFA = None
if not a_file.endswith('.yaml'):
    with open(a_file, 'r') as f:
        for line in f:
            elt = int(line)
            A_data[elt] = count
            count += 1

    A_untiled = Tensor.fromUncompressed(["S"], A_data, name ="A")
    A_HFA = A_untiled.splitUniform(32, relativeCoords=False) # split S
    print("A untiled shape {}, tiled shape {}".format(A_untiled.getShape(), A_HFA.getShape()))
    
    # A_HFA.print()
    # A_HFA.dump("tiled_frontier.yaml")
else: # already in pretiled yaml
    A_HFA = Tensor.fromYAMLfile(a_file)
    A_HFA.setName("A")
print("reading tiled mtx from yaml")
t0 = time.time()
b_file = sys.argv[3]
B_HFA = Tensor.fromYAMLfile(b_file)
t1 = time.time() - t0
print("read B from yaml in {} s".format(t1))
B_HFA.print()
# output
Z_data = [[0], [0]]
print(B_HFA.getShape())
Z_HFA = Tensor.fromUncompressed(["D1", "D0"], Z_data, shape=[B_HFA.getShape()[0],
B_HFA.getShape()[2]], name = "Z")

"""
a_file = sys.argv[2]
b_file = sys.argv[3]
A_HFA = get_A_HFA(a_file)
B_HFA = get_B_HFA(b_file)
Z_HFA = get_Z_HFA(B_HFA.getShape())
print("A shape {}, B shape {}, Z shape {}".format(A_HFA.getShape(), B_HFA.getShape
(), Z_HFA.getShape()))
A_HFA.print()
B_HFA.print()
Z_HFA.print()

output_descriptor = frontier_descriptor

A_shape = [A_HFA.getShape()[0], 32]
myA = encodeSwoopTensorInFormat(A_HFA, frontier_descriptor, tensor_shape=A_shape, cache_size=4*32)
print()
t0 = time.time()
myB = encodeSwoopTensorInFormat(B_HFA, ["U", "U", "C", "U"], cache_size=256 * 32 * 4)
t1 = time.time() - t0
print()
print("encoded B in {} s".format(t1))
myZ = encodeSwoopTensorInFormat(Z_HFA, output_descriptor, cache_size=4 * 256)

a.setImplementations("root", myA[0])
a.setImplementations("K1", myA[1])
a.setImplementations("K0", myA[2])
b.setImplementations("root", myB[0])
b.setImplementations("N1", myB[1])
b.setImplementations("K1", myB[2])
b.setImplementations("N0", myB[3])
b.setImplementations("K0", myB[4])
z.setImplementations("root", myZ[0])
z.setImplementations("N1", myZ[1])
z.setImplementations("N0", myZ[2])

"""
for i in range(0, len(myB[3])):
    myB[3][i].printFiber()

for i in range(0, len(myB[4])):
    myB[4][i].printFiber()
"""
#evaluate(b_n0ss, 2)          # 0,          1,          x, 2,          3,          x, x
#evaluate(b_n0_handlesss, 3)  # 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x, x, x
#evaluate(b_n0_payloadsss, 3) # 0, 1, 2, x, 0, 1, 2, x, x, 

#evaluate(b_k0sss, 3)         # 0,          1,          2,          x, 3,          4,          5,          x, x, 6, 7, 8, x, 9, 10, 11, x, x, x
#evaluate(b_k0_handlessss, 4) # 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, x
#evaluate(b_k0_coordssss, 4)   # 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x

#evaluate(b_valuessss, 4)
#exit(0)

stats_dict = dict()
cache_dict = dict()
evaluate(z_n0_update_acksss, 3, stats_dict=cache_dict)
evaluate(z_n1_update_acks, 1, stats_dict=cache_dict)
evaluate(z_root_update_ack, 0, stats_dict=cache_dict)

dumpAllStatsFromTensor(myA, stats_dict, cache_dict, 'A')
dumpAllStatsFromTensor(myB, stats_dict, cache_dict, 'B')
dumpAllStatsFromTensor(myZ, stats_dict, cache_dict, 'Z')

outpath = get_stats_dir(a_file, b_file)
# correctness testing
verifyInHFA(A_HFA, B_HFA, Z_HFA)

output_ref = compress_HFA_payloads(Z_HFA)
output_lin = get_lin_codec(myZ)

if output_lin is not output_ref:
    print(str_desc)
    print("codec: {}".format(output_lin))
    print("ref: {}".format(output_ref))
if len(output_ref[0]) == 0:
    for a in output_lin:
        assert(len(a) == 0)
else:
    assert(output_lin == output_ref)

# dump stats
with open(outpath + 'stats_' + str_desc, 'w') as statsfile:
    yaml.dump(stats_dict, statsfile)

with open(outpath + 'cache_'+ str_desc, 'w') as cachefile:
    yaml.dump(cache_dict, cachefile)


