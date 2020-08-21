from swoop import *



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

a = Tensor(name="A", rank_ids=["K1", "K0"])
b = Tensor(name="B", rank_ids=["K1", "N1", "K0", "N0"])
z = Tensor(name="Z", rank_ids=["N1", "N0"])

a_root = a["root"]
a_k1 = a["K1"]
a_k0 = a["K0"]
b_root = b["root"]
b_k1 = b["K1"]
b_n1 = b["N1"]
b_k0 = b["K0"]
b_n0 = b["N0"]
z_root = z["root"]
z_n1 = z["N1"]
z_n0 = z["N0"]

a_k1_fiber_handle = GetStartingFiber(a)
b_k1_fiber_handle = GetStartingFiber(b)
z_n1_fiber_handle = GetStartingFiber(z)

# a_k1 & b_k1
a_k1_handles = Scan(a_k1, a_k1_fiber_handle)
b_k1_handles = Scan(b_k1, b_k1_fiber_handle)
a_k1_coords = HandlesToCoords(a_k1, a_k1_handles)
b_k1_coords = HandlesToCoords(b_k1, b_k1_handles)
(ab_k1_coords, ab_a_k1_handles, ab_b_k1_handles) = Intersect(a_k1_coords, a_k1_handles, b_k1_coords, b_k1_handles, instance_name="K1")
ab_a_k1_payloads = HandlesToPayloads(a_k1, ab_a_k1_handles)
ab_b_k1_payloads = HandlesToPayloads(b_k1, ab_b_k1_handles)
a_k0_fiber_handles = PayloadsToFiberHandles(a_k1, ab_a_k1_payloads)
b_n1_fiber_handles = PayloadsToFiberHandles(b_k1, ab_b_k1_payloads)


# z_n1 << b_n1
b_n1_handless = Scan(b_n1, b_n1_fiber_handles)
b_n1_coordss = HandlesToCoords(b_n1, b_n1_handless)
b_n1_payloadss = HandlesToPayloads(b_n1, b_n1_handless)
# Repeat z_n1 iteration for each b_n1
z_n1_fiber_handles = Amplify(z_n1_fiber_handle, b_n1_fiber_handles)
(z_n1_handless, z_n1_new_fiber_handles) = InsertionScan(z_n1, z_n1_fiber_handles, b_n1_coordss)
z_n1_payloadss = HandlesToPayloads(z_n1, z_n1_handless)
b_k0_fiber_handless = PayloadsToFiberHandles(b_n1, b_n1_payloadss)
z_n0_fiber_handless = PayloadsToFiberHandles(z_n1, z_n1_payloadss)


# a_k0 & b_k0
b_k0_handlesss = Scan(b_k0, b_k0_fiber_handless)
# Repeat a_k0 iteration for each b_k0
a_k0_fiber_handless = Amplify(a_k0_fiber_handles, b_k0_fiber_handless, instance_name="K0")
a_k0_handlesss = Scan(a_k0, a_k0_fiber_handless)
a_k0_coordsss = HandlesToCoords(a_k0, a_k0_handlesss)
b_k0_coordsss = HandlesToCoords(b_k0, b_k0_handlesss)
(ab_k0_coordsss, ab_a_k0_handlesss, ab_b_k0_handlesss) = Intersect(a_k0_coordsss, a_k0_handlesss, b_k0_coordsss, b_k0_handlesss, instance_name="K0")
ab_a_k0_payloadsss = HandlesToPayloads(a_k0, ab_a_k0_handlesss)
ab_b_k0_payloadsss = HandlesToPayloads(b_k0, ab_b_k0_handlesss)
a_valuesss = PayloadsToValues(a_k0, ab_a_k0_payloadsss)
b_n0_fiber_handlesss = PayloadsToFiberHandles(b_k0, ab_b_k0_payloadsss)


# z_n0 << b_n0
b_n0_handlessss = Scan(b_n0, b_n0_fiber_handlesss)
b_n0_coordssss = HandlesToCoords(b_n0, b_n0_handlessss)
b_n0_payloadssss = HandlesToPayloads(b_n0, b_n0_handlessss)
# Repeat z_n0 iteration for each b_n0
z_n0_fiber_handlesss = Amplify(z_n0_fiber_handless, b_n0_fiber_handlesss, instance_name="N0")
(z_n0_handlessss, z_n0_new_fiber_handlesss) = InsertionScan(z_n0, z_n0_fiber_handlesss, b_n0_coordssss)
z_n0_payloadssss = HandlesToPayloads(z_n0, z_n0_handlessss)
a_valuessss = Amplify(a_valuesss, b_n0_handlessss)
b_valuessss = PayloadsToValues(b_n0, b_n0_payloadssss)
z_valuessss = PayloadsToValues(z_n0, z_n0_payloadssss)

# z_ref += a_val * b_val
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val, z_val: z_val + a_val * b_val
resultssss = Compute(body_func, a_valuessss, b_valuessss, z_valuessss)
# Reduce into the same value until end of rank
z_n0_update_ackssss = UpdatePayloads(z_n0, z_n0_handlessss, resultssss)

# Update N0 occupancy by summing all fiber occupancy.
z_n0_new_fiber_handless = Reduce(z_n0_new_fiber_handlesss)
z_n1_update_ackss = UpdatePayloads(z_n1, z_n1_handless, z_n0_new_fiber_handless)

# Update root occupancy
z_root_handle = Iterate(z_root)
z_root_handles = Amplify(z_root_handle, z_n1_new_fiber_handles)
z_root_update_acks = UpdatePayloads(z_root, z_root_handles, z_n1_new_fiber_handles)


N1 = 2
N0 = 3

K1 = 2
K0 = 3

my_a_root = BasicIntermediateRankImplementation(1, 1)
my_a_k1 = BasicIntermediateRankImplementation(K1, K0)
my_a_k0 = [BasicFiberImplementation([1, 2, 3]), BasicFiberImplementation([2, 4, 6])]
my_b_root = BasicIntermediateRankImplementation(1, 1)
my_b_k1 = BasicIntermediateRankImplementation(K1, N1)
my_b_n1 = [BasicIntermediateRankImplementation(N1, K0), BasicIntermediateRankImplementation(N1, K0, 1)]
my_b_k0 = [BasicIntermediateRankImplementation(K0, N0), BasicIntermediateRankImplementation(K0, N0, 1), BasicIntermediateRankImplementation(K0, N0, 2), BasicIntermediateRankImplementation(K0, N0, 3)]
my_b_n0 = [BasicFiberImplementation([4, 5,  6]), 
          BasicFiberImplementation([5,  6,  7]), 
          BasicFiberImplementation([6,  7,  8]),
          BasicFiberImplementation([12, 15, 18]), 
          BasicFiberImplementation([15, 18, 21]), 
          BasicFiberImplementation([18, 21, 24]),
          BasicFiberImplementation([8,  10, 12]), 
          BasicFiberImplementation([10, 12, 14]), 
          BasicFiberImplementation([12, 14, 16]),
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
b.setImplementations("K1", [my_b_k1])
b.setImplementations("N1", my_b_n1)
b.setImplementations("K0", my_b_k0)
b.setImplementations("N0", my_b_n0)
z.setImplementations("root", [my_z_root])
z.setImplementations("N1", [my_z_n1])
z.setImplementations("N0", my_z_n0)

evaluate(z_n0_update_ackssss, 4)
evaluate(z_n1_update_ackss, 2)
evaluate(z_root_update_acks, 1)

expected_vals = [[160, 190, 220], [352, 418, 484]]

print(f"Final K-Stationary result:")
for n1 in range(N1):
  print(my_z_n0[n1].vals)
for n1 in range(N1):
  assert(my_z_n0[n1].vals == expected_vals[n1])
print("==========================")
