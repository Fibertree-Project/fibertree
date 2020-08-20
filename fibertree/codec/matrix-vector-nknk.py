from swoop import *



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

a = Tensor(name="A", rank_ids=["K1", "K0"])
b = Tensor(name="B", rank_ids=["N1", "K1", "N0", "K0"])
z = Tensor(name="Z", rank_ids=["N1", "N0"])

a_root = a["root"]
a_k1 = a["K1"]
a_k0 = a["K0"]
b_root = b["root"]
b_n1 = b["N1"]
b_k1 = b["K1"]
b_n0 = b["N0"]
b_k0 = b["K0"]
z_root = z["root"]
z_n1 = z["N1"]
z_n0 = z["N0"]

a_k1_fiber_handle = GetStartingFiber(a)
b_n1_fiber_handle = GetStartingFiber(b)
z_n1_fiber_handle = GetStartingFiber(z)

# z_n1 << b_n1
b_n1_handles = Scan(b_n1, b_n1_fiber_handle)
b_n1_coords = HandlesToCoords(b_n1, b_n1_handles)
b_n1_payloads = HandlesToPayloads(b_n1, b_n1_handles)
(z_n1_handles, z_n1_new_fiber_handle) = InsertionScan(z_n1, z_n1_fiber_handle, b_n1_coords)
z_n1_payloads = HandlesToPayloads(z_n1, z_n1_handles)
b_k1_fiber_handles = PayloadsToFiberHandles(b_n1, b_n1_payloads)
z_n0_fiber_handles = PayloadsToFiberHandles(z_n1, z_n1_payloads)


# a_k1 & b_k1
b_k1_handless = Scan(b_k1, b_k1_fiber_handles)
# Repeat a_k1 iteration for each b_k1
a_k1_fiber_handles = Amplify(a_k1_fiber_handle, b_k1_fiber_handles, instance_name="K1")
a_k1_handless = Scan(a_k1, a_k1_fiber_handles) 
a_k1_coordss = HandlesToCoords(a_k1, a_k1_handless)
b_k1_coordss = HandlesToCoords(b_k1, b_k1_handless)
(ab_k1_coordss, ab_a_k1_handless, ab_b_k1_handless) = Intersect(a_k1_coordss, a_k1_handless, b_k1_coordss, b_k1_handless, instance_name="K1")
ab_a_k1_payloadss = HandlesToPayloads(a_k1, ab_a_k1_handless)
ab_b_k1_payloadss = HandlesToPayloads(b_k1, ab_b_k1_handless)
a_k0_fiber_handless = PayloadsToFiberHandles(a_k1, ab_a_k1_payloadss)
b_n0_fiber_handless = PayloadsToFiberHandles(b_k1, ab_b_k1_payloadss)

# z_n0 << b_n0
b_n0_handlesss = Scan(b_n0, b_n0_fiber_handless)
b_n0_coordsss = HandlesToCoords(b_n0, b_n0_handlesss)
b_n0_payloadsss = HandlesToPayloads(b_n0, b_n0_handlesss)
# Repeat z_n0 iteration for each b_n0
z_n0_fiber_handless = Amplify(z_n0_fiber_handles, b_n0_fiber_handless, instance_name="N0")
(z_n0_handlesss, z_n0_new_fiber_handless) = InsertionScan(z_n0, z_n0_fiber_handless, b_n0_coordsss)
z_n0_payloadsss = HandlesToPayloads(z_n0, z_n0_handlesss)
b_k0_fiber_handlesss = PayloadsToFiberHandles(b_n0, b_n0_payloadsss)
z_valuesss = PayloadsToValues(z_n0, z_n0_payloadsss)

# a_k0 & b_k0
b_k0_handlessss = Scan(b_k0, b_k0_fiber_handlesss)
# Repeat a_k0 iteration for each b_k0
a_k0_fiber_handlesss = Amplify(a_k0_fiber_handless, b_k0_fiber_handlesss, instance_name="K0")
a_k0_handlessss = Scan(a_k0, a_k0_fiber_handlesss)
a_k0_coordssss = HandlesToCoords(a_k0, a_k0_handlessss)
b_k0_coordssss = HandlesToCoords(b_k0, b_k0_handlessss)
(ab_k0_coordssss, ab_a_k0_handlessss, ab_b_k0_handlessss) = Intersect(a_k0_coordssss, a_k0_handlessss, b_k0_coordssss, b_k0_handlessss, instance_name="K0")
ab_a_k0_payloadssss = HandlesToPayloads(a_k0, ab_a_k0_handlessss)
ab_b_k0_payloadssss = HandlesToPayloads(b_k0, ab_b_k0_handlessss)
a_valuessss = PayloadsToValues(a_k0, ab_a_k0_payloadssss)
b_valuessss = PayloadsToValues(b_k0, ab_b_k0_payloadssss)


# z_ref += a_val * b_val
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val: a_val * b_val
partial_productssss = Compute(body_func, a_valuessss, b_valuessss)
# Reduce into the same value until end of rank
#z_new_valuesss = Reduce(partial_productssss, z_valuesss, instance_name="K0")
z_new_valuesss = Reduce(partial_productssss, instance_name="K0")
z_n0_update_acksss = UpdatePayloads(z_n0, z_n0_handlesss, z_new_valuesss)

# Update N0 occupancy by summing all fiber occupancy.
z_n1_handless = Amplify(z_n1_handles, b_n0_fiber_handless, instance_name="N1_Upd")
z_n1_update_acks = UpdatePayloads(z_n1, z_n1_handless, z_n0_new_fiber_handless)

# Update root occupancy
z_root_handle = Iterate(z_root)
z_root_update_ack = UpdatePayloads(z_root, z_root_handle, z_n1_new_fiber_handle)


N1 = 2
N0 = 3

K1 = 2
K0 = 3

my_a_root = BasicIntermediateRankImplementation(1, 1)
my_a_k1 = BasicIntermediateRankImplementation(K1, K0)
my_a_k0 = [BasicFiberImplementation([1, 2, 3]), BasicFiberImplementation([2, 4, 6])]
my_b_root = BasicIntermediateRankImplementation(1, 1)
my_b_n1 = BasicIntermediateRankImplementation(N1, K1)
my_b_k1 = [BasicIntermediateRankImplementation(K1, N0), BasicIntermediateRankImplementation(K1, N0)]
my_b_n0 = [BasicIntermediateRankImplementation(N0, K0), BasicIntermediateRankImplementation(N0, K0), BasicIntermediateRankImplementation(N0, K0), BasicIntermediateRankImplementation(N0, K0)]
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
          BasicFiberImplementation([20, 24, 26]), 
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

evaluate(z_n0_update_acksss, 3)
evaluate(z_n1_update_acks, 1)
evaluate(z_root_update_ack, 0)

print(f"Final Z-Stationary result:")
for n1 in range(N1):
  print(my_z_n0[n1].vals)
#for n1 in range(N1):
#  assert(my_z_n0[n1].vals == expected_vals[n1])
print("==========================")
