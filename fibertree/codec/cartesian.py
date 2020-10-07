from swoop import *


## Test program: Parallelized Cartesian product
#
#  Z_mn = A_m * B_n
# Tiled:
#  Z_n1mn0 = Am * Bn1n0      #  n0 is parallel
#
# Option 1: 
#  B_NN = B_N.splitUniform(N0)
#  Z_NMN = Z_MN.splitUniform(N0).swapRanks()
# 
# Option 2: 
#  B_NN = B_N.splitEqual(N0)
#  Z_NMN = Z_MN.splitNonUniform(B_NN.getRoot().getCoords()).swapRanks()
#
#  for n1, (z_m, b_n0) in z_n1 << b_n1:
#    for m, (z_n0, a_val) in z_m << a_m:
#      parallel for n0, (z_ref, b_val) in z_n0 << b_n0:
#        z_ref <<= a_val * b_val
#


a = SwoopTensor(name="A", rank_ids=["M"])
b = SwoopTensor(name="B", rank_ids=["N1", "N0"])
z = SwoopTensor(name="Z", rank_ids=["N1", "M", "N0"])

a_m = a.getStartHandle()
b_n1 = b.getStartHandle()
z_root = z.getRootHandle()
z_n1 = z.getStartHandle()

# n1 << operator, RHS:
b_n1_handles = Scan(b_n1)
b_n1_coords = HandlesToCoords(b_n1, b_n1_handles)
b_n1_payloads = HandlesToPayloads(b_n1, b_n1_handles)
b_n0s = PayloadsToFiberHandles(b_n1, b_n1_payloads)
# n1 << operator:
(z_n1_handles, z_n1_updated_fiber_handle) = InsertionScan(z_n1, b_n1_coords)
z_n1_payloads = HandlesToPayloads(z_n1, z_n1_handles)
z_ms = PayloadsToFiberHandles(z_n1, z_n1_payloads)

# m << operator, RHS, repeated b_n1 more times:
a_ms = Amplify(a_m, b_n1_handles)
a_m_handless = Scan(a_ms)
a_m_coordss = HandlesToCoords(a_ms, a_m_handless)
a_m_payloadss = HandlesToPayloads(a_ms, a_m_handless)
a_valuess = PayloadsToValues(a_ms, a_m_payloadss)
# m << operator:
(z_m_handless, z_m_updated_fiber_handles) = InsertionScan(z_ms, a_m_coordss)
z_m_payloadss = HandlesToPayloads(z_ms, z_m_handless)
z_n0ss = PayloadsToFiberHandles(z_ms, z_m_payloadss)

# n0 << operator, RHS, repeated a_m more times:
b_n0ss = Amplify(b_n0s, a_m_handless, instance_name="B_N0")
b_n0_handlesss = Scan(b_n0ss)
b_n0_coordsss = HandlesToCoords(b_n0ss, b_n0_handlesss)
b_n0_payloadsss = HandlesToPayloads(b_n0ss, b_n0_handlesss)
a_valuesss = Amplify(a_valuess, b_n0_handlesss, instance_name="A_N0")
b_valuesss = PayloadsToValues(b_n0ss, b_n0_payloadsss)
# n0 << operator:
(z_n0_handlesss, z_n0_updated_fiber_handless) = InsertionScan(z_n0ss, b_n0_coordsss)
# z_values not referenced in loop body, so don't retrieve it

# z_ref <<= a_val * b_val
# Original sequential code
resultsss = Compute(lambda a, b: a * b, a_valuesss, b_valuesss)

# BEGIN PARALLEL_FOR
#NUM_PES = 4
#dist_func = lambda n: n % NUM_PES 
#n0_distribution_choices = Compute(dist_func, b_n0_coords)
#b_values_distributed = Distribute(4, n0_distribution_choices, b_values)
#body_func = lambda a_val, b_val: a_val * b_val
#results = []
#for n0 in range(NUM_PES):
#  results.append(Compute(body_func, a_values, b_values_distributed[n0], instance_name=str(n0)))
  
#resultsss = Collect(NUM_PES, n0_distribution_choices, results)
# END PARALLEL FOR

# n0 << operator, LHS:
z_n0_acksss = UpdatePayloads(z_n0ss, z_n0_handlesss, resultsss)

# m << operator, LHS:
z_m_ackss = UpdatePayloads(z_ms, z_m_handless, z_n0_updated_fiber_handless)

# n1 << operator, LHS:
z_n1_acks = UpdatePayloads(z_n1, z_n1_handles, z_m_updated_fiber_handles)
z_root_ack = UpdatePayloads(z_root, Stream0(0), z_n1_updated_fiber_handle)

M = 3
N1 = 1
N0 = 3
my_a_root = BasicIntermediateRankImplementation(1, 1)
my_a_m = BasicFiberImplementation([1, 2, 3])
my_b_root = BasicIntermediateRankImplementation(1, 1)
my_b_n1 = BasicIntermediateRankImplementation(N1, N0)
my_b_n0 = [BasicFiberImplementation([4, 5, 6])]
my_z_root = BasicIntermediateRankImplementation(1, 1)
my_z_n1 = BasicIntermediateRankImplementation(N1, M)
my_z_m  = BasicIntermediateRankImplementation(M, N0)
my_z_n0 = []
for m in range(M):
  for n1 in range(N1):
    my_z_n0.append(BasicFiberImplementation([0] * N0))


a.setImplementations("root", [my_a_root])
a.setImplementations("M", [my_a_m])
b.setImplementations("root", [my_b_root])
b.setImplementations("N1", [my_b_n1])
b.setImplementations("N0", my_b_n0)
z.setImplementations("root", [my_z_root])
z.setImplementations("N1", [my_z_n1])
z.setImplementations("M", [my_z_m])
z.setImplementations("N0", my_z_n0)


evaluate(z_n0_acksss, 3)
evaluate(z_m_ackss, 2)
evaluate(z_n1_acks, 1)
evaluate(z_root_ack, 0)

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
