from swoop import *
from swoop_util import *
from fibertree import Tensor
import sys

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
# was here before
# z_n1_update_ackss = UpdatePayloads(z_n1ss, z_n1_handlesss, z_n0_new_fiber_handlesss)

# below here is new test
zeros = Amplify(Stream0(0), z_n1s)
zeross = Amplify(zeros, z_n1_handless)
z_n0_new_fiber_handless = Reduce(z_n0_new_fiber_handlesss, zeross)
z_n1_update_ackss = UpdatePayloads(z_n1s, z_n1_handless, z_n0_new_fiber_handless)
# Update root occupancy
z_root_handles = Amplify(Stream0(0), z_n1_new_fiber_handles)
z_root_update_acks = UpdatePayloads(z_root, z_root_handles, z_n1_new_fiber_handles)


N1 = 2
N0 = 3

K1 = 2
K0 = 3

A_data = [[1, 2, 3], [2, 4, 6]]
B_data = [
        [[[4, 5,  6],
          [5,  6,  7],
          [6,  7,  8]],
          [[12, 15, 18],
          [15, 18, 21],
          [18, 21, 24]]],
          [[[8,  10, 12],
          [10, 12, 14],
          [12, 14, 16]],
          [[16, 20, 24],
          [20, 24, 28],
          [24, 28, 32]]]
          ]

Z_data = [[0, 0, 0], [0, 0, 0]]

A_HFA = Tensor.fromUncompressed(["K1", "K0"], A_data, name = "A")
B_HFA = Tensor.fromUncompressed(["K1", "N1", "K0", "N0"], B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(["N1", "N0"], Z_data, name = "Z")

str_desc = sys.argv[1]
# output_desc = sys.argv[2]
frontier_descriptor = [str_desc[0], str_desc[1]]
output_descriptor = frontier_descriptor
# output_descriptor = [output_desc[0], output_desc[1]]

myA = encodeSwoopTensorInFormat(A_HFA, frontier_descriptor)
myB = encodeSwoopTensorInFormat(B_HFA, ["U", "U", "U", "C"])
myZ = encodeSwoopTensorInFormat(Z_HFA, output_descriptor)
print(myA)
print(myB)
print(myZ)

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

evaluate(z_n0_update_ackssss, 4)
evaluate(z_n1_update_ackss, 2)
evaluate(z_root_update_acks, 1)

expected_vals = [[160, 190, 220], [352, 418, 484]]

print("Z: {}".format(myZ))
output_for_check = [myZ[2][0].getPayloads(), myZ[2][1].getPayloads()]
print(output_for_check)
assert(output_for_check == expected_vals)
"""
print(f"Final K-Stationary result:")
for n1 in range(N1):
  print(my_z_n0[n1].vals)
for n1 in range(N1):
  assert(my_z_n0[n1].vals == expected_vals[n1])
print("==========================")
"""
