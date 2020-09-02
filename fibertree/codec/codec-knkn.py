from swoop import *
from swoop_util import *
from fibertree import Tensor
import sys
import time
## Test program: Tiled K-Stationary vector-matrix multiplication
#
#   Z_n = A_k * B_kn
# Tiled:
#   Z_n1n0 = A_k1k0 * B_k1n1k0n0
#
#for k1, (a_k0, b_n1) in a_k1 & b_k1:
#  for n1, (z_n0, b_n0) in z_n1 << b_n1:
#    for k0, (a, b_n0) in a_k0 & b_k0:
#      for n0, (z, b) in z_n0 << b_n0:
#        z += a * b

a = SwoopTensor(name="A", rank_ids=["K1", "K0"])
b = SwoopTensor(name="B", rank_ids=["K1", "N1", "K0", "N0"])
z = SwoopTensor(name="Z", rank_ids=["N1", "N0"])


a_k1 = a.getStartHandle()
b_k1 = b.getStartHandle()
z_n1 = z.getStartHandle()
z_root = z.getRootHandle()

# a_k1 & b_k1
a_k1_handles = Scan(a_k1)
b_k1_handles = Scan(b_k1)
a_k1_coords = HandlesToCoords(a_k1, a_k1_handles)
b_k1_coords = HandlesToCoords(b_k1, b_k1_handles)
(ab_k1_coords, ab_a_k1_handles, ab_b_k1_handles) = Intersect(a_k1_coords, a_k1_handles, b_k1_coords, b_k1_handles, instance_name="K1")
ab_a_k1_payloads = HandlesToPayloads(a_k1, ab_a_k1_handles)
ab_b_k1_payloads = HandlesToPayloads(b_k1, ab_b_k1_handles)
a_k0s = PayloadsToFiberHandles(a_k1, ab_a_k1_payloads)
b_n1s = PayloadsToFiberHandles(b_k1, ab_b_k1_payloads)


# z_n1 << b_n1
b_n1_handless = Scan(b_n1s)
b_n1_coordss = HandlesToCoords(b_n1s, b_n1_handless)
b_n1_payloadss = HandlesToPayloads(b_n1s, b_n1_handless)
# Repeat z_n1 iteration for each b_n1
z_n1s = Amplify(z_n1, b_n1s)
(z_n1_handless, z_n1_new_fiber_handles) = InsertionScan(z_n1s, b_n1_coordss)
z_n1_payloadss = HandlesToPayloads(z_n1s, z_n1_handless)
b_k0ss = PayloadsToFiberHandles(b_n1s, b_n1_payloadss)
z_n0ss = PayloadsToFiberHandles(z_n1s, z_n1_payloadss)


# a_k0 & b_k0
b_k0_handlesss = Scan(b_k0ss)
# Repeat a_k0 iteration for each b_k0
a_k0ss = Amplify(a_k0s, b_k0ss, instance_name="K0")
a_k0_handlesss = Scan(a_k0ss)
a_k0_coordsss = HandlesToCoords(a_k0ss, a_k0_handlesss)
b_k0_coordsss = HandlesToCoords(b_k0ss, b_k0_handlesss)
(ab_k0_coordsss, ab_a_k0_handlesss, ab_b_k0_handlesss) = Intersect(a_k0_coordsss, a_k0_handlesss, b_k0_coordsss, b_k0_handlesss, instance_name="K0")
ab_a_k0_payloadsss = HandlesToPayloads(a_k0ss, ab_a_k0_handlesss)
ab_b_k0_payloadsss = HandlesToPayloads(b_k0ss, ab_b_k0_handlesss)
a_valuesss = PayloadsToValues(a_k0ss, ab_a_k0_payloadsss)
b_n0sss = PayloadsToFiberHandles(b_k0ss, ab_b_k0_payloadsss)


# z_n0 << b_n0
b_n0_handlessss = Scan(b_n0sss)
b_n0_coordssss = HandlesToCoords(b_n0sss, b_n0_handlessss)
b_n0_payloadssss = HandlesToPayloads(b_n0sss, b_n0_handlessss)
# Repeat z_n0 iteration for each b_n0
z_n0sss = Amplify(z_n0ss, b_n0sss, instance_name="N0")
(z_n0_handlessss, z_n0_new_fiber_handlesss) = InsertionScan(z_n0sss, b_n0_coordssss)
z_n0_payloadssss = HandlesToPayloads(z_n0sss, z_n0_handlessss)
a_valuessss = Amplify(a_valuesss, b_n0_handlessss)
b_valuessss = PayloadsToValues(b_n0sss, b_n0_payloadssss)
z_valuessss = PayloadsToValues(z_n0sss, z_n0_payloadssss)

# z_ref += a_val * b_val
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val, z_val: z_val + a_val * b_val
resultssss = Compute(body_func, a_valuessss, b_valuessss, z_valuessss)
# Reduce into the same value until end of rank
z_n0_update_ackssss = UpdatePayloads(z_n0sss, z_n0_handlessss, resultssss)

# Update N0 occupancy. (Should we be reducing here?)

z_n1s = Amplify(z_n1, b_n1s)
z_n1ss = Amplify(z_n1s, b_k0ss)
z_n1_handlesss = Amplify(z_n1_handless, b_n0sss)
zeros = Amplify(Stream0(0), z_n1s)
zeross = Amplify(zeros, z_n1_handless)
# final_resultsss = Reduce(resultssss, zerosss)
z_n0_new_fiber_handless = Reduce(z_n0_new_fiber_handlesss, zeross)
# z_n1_update_acksss = UpdatePayloads(z_n1ss, z_n1_handlesss, z_n0_new_fiber_handlesss)
z_n1_update_ackss = UpdatePayloads(z_n1s, z_n1_handless, z_n0_new_fiber_handless)
# Update root occupancy
z_root_handles = Amplify(Stream0(0), z_n1_new_fiber_handles)
z_root_update_acks = UpdatePayloads(z_root, z_root_handles, z_n1_new_fiber_handles)

# read in inputs
# jhu_len = 5157
shape = 500 # TODO: take this as input or get it from yaml somehow
# generate input frontier and tile it
A_data = [0] * shape

# read in frontier
with open(sys.argv[2], 'r') as f:
    for line in f:
        elt = int(line)
        A_data[elt] = 1

A_untiled = Tensor.fromUncompressed(["S"], A_data, name = "A")
A_HFA = A_untiled.splitUniform(32, relativeCoords=False) # split S
print("A untiled shape {}, tiled shape {}".format(A_untiled.getShape(), A_HFA.getShape()))
# A_HFA.dump("tiled_frontier.yaml")


print("reading tiled mtx from yaml")
t0 = time.clock()
B_HFA = Tensor.fromYAMLfile(sys.argv[3])
t1 = time.clock() - t0
print("read B from yaml in {} s".format(t1))

# output
Z_data = [[0], [0]]
Z_HFA = Tensor.fromUncompressed(["D1", "D0"], Z_data, shape=[B_HFA.getShape()[1], B_HFA.getShape()[3]], name = "Z")
# Z_HFA = Z_untiled.splitUniform(256) # split D
# print("A shape {}, Z shape {}".format(A_HFA.getShape(), Z_HFA.getShape()))
print("A shape {}, B shape {}, Z shape {}".format(A_HFA.getShape(), B_HFA.getShape(), Z_HFA.getShape()))

# exit(0)
str_desc = sys.argv[1]
# output_desc = sys.argv[2]
frontier_descriptor = [str_desc[0], str_desc[1]]
output_descriptor = frontier_descriptor
# output_descriptor = [output_desc[0], output_desc[1]]

myA = encodeSwoopTensorInFormat(A_HFA, frontier_descriptor)
t0 = time.clock()
myB = encodeSwoopTensorInFormat(B_HFA, ["U", "U", "U", "C"])
t1 = time.clock() - t0
print("encoded B in {} s".format(t1))

myZ = encodeSwoopTensorInFormat(Z_HFA, output_descriptor)

a.setImplementations("root", myA[0])
a.setImplementations("K1", myA[1])
a.setImplementations("K0", myA[2])
b.setImplementations("root", myB[0])
b.setImplementations("K1", myB[1])
b.setImplementations("N1", myB[2])
b.setImplementations("K0", myB[3])
b.setImplementations("N0", myB[4])
z.setImplementations("root", myZ[0])
z.setImplementations("N1", myZ[1])
z.setImplementations("N0", myZ[2])

# print("Z[1]: {}".format(myZ[1]))
# print("Z[2]: {}".format(myZ[2]))

# print("len B[0] {}, B[1] {}, B[2] {}, B[3] {}, B[4] {}".format(len(myB[0]), len(myB[1]), len(myB[2]), len(myB[3]), len(myB[4])))
#for i in range(0, len(myB[3])):
#    myB[3][i].printFiber()

# exit(0)

evaluate(z_n0_update_ackssss, 4)
evaluate(z_n1_update_ackss, 2)
evaluate(z_root_update_acks, 1)

# do verification
a_k1 = A_HFA.getRoot()
z_n1 = Z_HFA.getRoot()
b_k1 = B_HFA.getRoot()
for k1, (a_k0, b_n1) in a_k1 & b_k1:
  for n1, (z_n0, b_k0) in z_n1 << b_n1:
    for k0, (a, b_n0) in a_k0 & b_k0:
      for n0, (z, b) in z_n0 << b_n0:
        z += a * b
Z_HFA.print()

output_lin = []
myZ[1][0].printFiber()
for i in range(0, len(myZ[2])):
    myZ[2][i].printFiber()
    output_lin.append(myZ[2][i].getPayloads())

z_n1 = Z_HFA.getRoot()
output_ref = []
for (z, z_n0) in z_n1:
    output_ref.append(z_n0.getPayloads())

assert(output_lin == output_ref)
