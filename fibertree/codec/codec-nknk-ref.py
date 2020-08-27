from swoop import *
from swoop_util import *
from fibertree import Tensor
import sys
import yaml

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


N1 = 2
N0 = 3

K1 = 2
K0 = 3

A_data = [[1, 2, 3], [2, 4, 6]]
B_data = [
[[[4, 5,  6],
[5,  6,  7],
[6,  7,  8]],
[[8,  10, 12],
[10, 12, 14],
[12, 14, 16]]],
[[[12, 15, 18],
[15, 18, 21],
[18, 21, 24]],
[[16, 20, 24],
[20, 24, 28],
[24, 28, 32]]]
]

Z_data = [[0, 0, 0], [0, 0, 0]]

A_HFA = Tensor.fromUncompressed(["K1", "K0"], A_data, name = "A")
B_HFA = Tensor.fromUncompressed(["N1", "K1", "N0", "K0"], B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(["N1", "N0"], Z_data, shape=[N1, N0], name = "Z")

str_desc = sys.argv[1]
frontier_descriptor = [str_desc[0], str_desc[1]]

myA = encodeSwoopTensorInFormat(A_HFA, frontier_descriptor)
print("encoded A")
myB = encodeSwoopTensorInFormat(B_HFA, ["U", "U", "U", "C"])
print("encoded B")
myZ = encodeSwoopTensorInFormat(Z_HFA, frontier_descriptor)
print("done encoding\n")

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
my_a_root = BasicIntermediateRankImplementation(1, 1)
my_a_k1 = BasicIntermediateRankImplementation(K1, K0)
my_a_k0 = [BasicFiberImplementation([1, 2, 3]), BasicFiberImplementation([2, 4, 6])]
my_b_root = BasicIntermediateRankImplementation(1, 1)
my_b_n1 = BasicIntermediateRankImplementation(N1, K1)
my_b_k1 = [BasicIntermediateRankImplementation(K1, N0), BasicIntermediateRankImplementation(K1, N0, 1)]
my_b_n0 = [BasicIntermediateRankImplementation(N0, K0), BasicIntermediateRankImplementation(N0, K0, 1), BasicIntermediateRankImplementation(N0, K0, 2), BasicIntermediateRankImplementation(N0, K0, 3)]
my_b_k0 = [BasicFiberImplementation([4, 5,  6]), 
          BasicFiberImplementation([5,  6,  7]), 
          BasicFiberImplementation([6,  7,  8]),
          BasicFiberImplementation([8,  10, 12]), 
          BasicFiberImplementation([10, 12, 14]), 
          BasicFiberImplementation([12, 14, 16]),
          BasicFiberImplementation([12, 15, 18]), 
          BasicFiberImplementation([15, 18, 21]), 
          BasicFiberImplementation([18, 21, 24]),
          BasicFiberImplementation([16, 20, 24]), 
          BasicFiberImplementation([20, 24, 28]), 
          BasicFiberImplementation([24, 28, 32])]

my_z_root = BasicIntermediateRankImplementation(1, 1)
my_z_n1 = BasicIntermediateRankImplementation(N1, N0)
my_z_n0 = []
for n1 in range(N1):
  my_z_n0.append(BasicFiberImplementation([0] * N0))


a.setImplementations("root", [my_a_root])
a.setImplementations("K1", [my_a_k1])
a.setImplementations("K0", my_a_k0)
b.setImplementations("root", [my_b_root])
b.setImplementations("N1", [my_b_n1])
b.setImplementations("K1", my_b_k1)
b.setImplementations("N0", my_b_n0)
b.setImplementations("K0", my_b_k0)
z.setImplementations("root", [my_z_root])
z.setImplementations("N1", [my_z_n1])
z.setImplementations("N0", my_z_n0)
"""
#evaluate(b_n0ss, 2)          # 0,          1,          x, 2,          3,          x, x
#evaluate(b_n0_handlesss, 3)  # 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x, x, x
#evaluate(b_n0_payloadsss, 3) # 0, 1, 2, x, 0, 1, 2, x, x, 

#evaluate(b_k0sss, 3)         # 0,          1,          2,          x, 3,          4,          5,          x, x, 6, 7, 8, x, 9, 10, 11, x, x, x
#evaluate(b_k0_handlessss, 4) # 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, x
#evaluate(b_k0_coordssss, 4)   # 0, 1, 2, x, 0, 1, 2, x, 0, 1, 2, x, x, 0, 1, 2, x, 0, 1, 2, x

#evaluate(b_valuessss, 4)
#exit(0)

evaluate(z_n0_update_acksss, 3)
evaluate(z_n1_update_acks, 1)
evaluate(z_root_update_ack, 0)

expected_vals = [[160, 190, 220], [352, 418, 484]]
# print("Z: {}".format(myZ))
# myZ[1][0].printFiber()
# myZ[1][0].payloads[0].printFiber()
# myZ[1][0].payloads[1].printFiber()

output_for_check = [myZ[2][0].getPayloads(), myZ[2][1].getPayloads()]
print(output_for_check)
assert(output_for_check == expected_vals)
"""
stats_dict = dict()
dumpAllStatsFromTensor(myA, stats_dict)
dumpAllStatsFromTensor(myB, stats_dict)
dumpAllStatsFromTensor(myZ, stats_dict)
# print("\nZ-stationary vector-matrix: A = <T>, B = <T, T>, Z = <T>")
print(yaml.dump(stats_dict))

print(f"Final Z-Stationary result:")
for n1 in range(N1):
  print(my_z_n0[n1].vals)
for n1 in range(N1):
  assert(my_z_n0[n1].vals == expected_vals[n1])
print("==========================")
"""
