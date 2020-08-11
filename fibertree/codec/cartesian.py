from swoop import *


## Test program: Parallelized Cartesian product
#
#  Z_mn = A_m * B_n
# Tiled:
#  Z_n1mn0 = Am * Bn1n0      #  n0 == numPEs
#
#  for n1, (z_m, b_n0) in z_n1 << b_n1:
#    for m, (z_n0, a_val) in z_m << a_m:
#      parallel for n0, (z_ref, b_val) in z_n0 << b_n0:
#        z_ref <<= a_val * b_val
#


a = Tensor(name="A", rank_ids=["M"])
b = Tensor(name="B", rank_ids=["N1", "N0"])
z = Tensor(name="Z", rank_ids=["N1", "M", "N0"])

a_m = a["M"]
b_n1 = b["N1"]
b_n0 = b["N0"]
z_root = z["root"]
z_n1 = z["N1"]
z_m = z["M"]
z_n0 = z["N0"]

# n1 << operator, RHS:
b_n1_handles = Iterate(b_n1)
b_n1_coords = HandlesToCoords(b_n1, b_n1_handles)
b_n1_payloads = HandlesToPayloads(b_n1, b_n1_handles)
b_n0_fiber_handles = PayloadsToFiberHandles(b_n1, b_n1_payloads)
# n1 << operator:
z_n1_fiber_handle = Stream0(z_n1.getStartingFiberHandle())
(z_n1_handles, z_n1_updated_fiber_handles) = InsertionScan(z_n1, z_n1_fiber_handle, b_n1_coords)
z_n1_coords = b_n1_coords
z_n1_payloads = HandlesToPayloads(z_n1, z_n1_handles)
z_m_fiber_handles = PayloadsToFiberHandles(z_n1, z_n1_payloads)

# m << operator, RHS, repeated b_n1 more times:
a_m_fiber_handle = Stream0(a_m.getStartingFiberHandle())
a_m_fiber_handles = Amplify(a_m_fiber_handle, b_n1_handles)
a_m_handles = Scan(a_m, a_m_fiber_handles)
### StartOfFiber: move to fiber[fiber_handle] and setupSlice
### SteadyState: nextInSlice until fiber end
a_m_coords = HandlesToCoords(a_m, a_m_handles)
a_m_payloads = HandlesToPayloads(a_m, a_m_handles)
a_values = PayloadsToValues(a_m, a_m_payloads)
# m << operator:
(z_m_handles, z_m_updated_fiber_handles) = InsertionScan(z_m, z_m_fiber_handles, a_m_coords)
z_m_coords = a_m_coords
z_m_payloads = HandlesToPayloads(z_m, z_m_handles)
z_n0_fiber_handles = PayloadsToFiberHandles(z_m, z_m_payloads)

# n0 << operator, RHS, repeated a_m more times:
b_n0_fiber_handles_amplified = Amplify(b_n0_fiber_handles, a_m_handles)
b_n0_handles = Scan(b_n0, b_n0_fiber_handles_amplified)
b_n0_coords = HandlesToCoords(b_n0, b_n0_handles)
b_n0_payloads = HandlesToPayloads(b_n0, b_n0_handles)
b_values = PayloadsToValues(b_n0, b_n0_payloads)
# n0 << operator:
(z_n0_handles, z_n0_updated_fiber_handles) = InsertionScan(z_n0, z_n0_fiber_handles, b_n0_coords)
z_n0_coords = b_n0_coords
# z_values not referenced in loop body, so don't retrieve it

# z_ref <<= a_val * b_val
a_values_amplified = Amplify(a_values, b_values)
body_func = lambda a_val, b_val: a_val * b_val
results = Compute(body_func, a_values_amplified, b_values)


# n0 << operator, LHS:
z_n0_acks = UpdatePayloads(z_n0, z_n0_handles, results)

# m << operator, LHS:
z_m_acks = UpdatePayloads(z_m, z_m_handles, z_n0_updated_fiber_handles)

# n1 << operator, LHS:
z_n1_acks = UpdatePayloads(z_n1, z_n1_handles, z_m_updated_fiber_handles)
z_root_handles = Iterate(z_root)
z_root_acks = UpdatePayloads(z_root, z_root_handles, z_n1_updated_fiber_handles)

M = 3
N1 = 1
N0 = 3
my_a_m = BasicFiberImplementation([1, 2, 3])
my_b_n1 = BasicIntermediateRankImplementation(N1, N0)
my_b_n0 = [BasicFiberImplementation([4, 5, 6])]
my_z_n1 = BasicIntermediateRankImplementation(N1, M)
my_z_m  = BasicIntermediateRankImplementation(M, N0)
my_z_n0 = []
for m in range(M):
  for n1 in range(N1):
    my_z_n0.append(BasicFiberImplementation([0] * N0))


a.setImplementations("M", [my_a_m])
b.setImplementations("N1", [my_b_n1])
b.setImplementations("N0", my_b_n0)
z.setImplementations("root", [BasicIntermediateRankImplementation(1, 1)])
z.setImplementations("N1", [my_z_n1])
z.setImplementations("M", [my_z_m])
z.setImplementations("N0", my_z_n0)


evaluate(z_n0_acks, 2)

expected_vals = [[4, 5, 6], [8, 10, 12], [12, 15, 18]]

print("==========================")
print(f"Final Z-Stationary result:")
for n1 in range(N1):
  for m in range(M):
    print(my_z_n0[n1 * M +  m].vals)
for n1 in range(N1):
  for m in range(M):
    assert(my_z_n0[n1 * M + m].vals == expected_vals[n1 * M + m])
print("==========================")
