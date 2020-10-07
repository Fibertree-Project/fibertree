from swoop import *


## Test program: Parallelized dot product
#
#  Z = Sum(k). A_k * B_k
# Tiled:
#  Z = Sum(k1).Sum(K0). A_k1k0 * Bk1k0      #  k0 is parallel
#
# Option 1: 
#  A_KK = A_K.splitUniform(K0)
#  B_KK = B_K.splitUniform(K0)
# 
# Option 2: 
#  A_KK = A_K.splitEqual(K0)
#  B_KK = B_K.splitNonUniform(A_KK.getRoot().getCoords()).swapRanks()
#
#  for k1, (a_k0, b_k0) in a_k1 & b_k1:
#      parallel for k0, (a_val, b_val) in a_k0 & b_k0:
#        z_ref <<= a_val * b_val
#


a = SwoopTensor(name="A", rank_ids=["K1", "K0"])
b = SwoopTensor(name="B", rank_ids=["K1", "K0"])
z = SwoopTensor(name="Z", rank_ids=[])

a_k1 = a.getStartHandle()
b_k1 = b.getStartHandle()
z_root = z.getRootHandle()


# k1 & operator:
a_k1_handles = Scan(a_k1)
b_k1_handles = Scan(b_k1)
a_k1_coords = HandlesToCoords(a_k1, a_k1_handles)
b_k1_coords = HandlesToCoords(b_k1, b_k1_handles)
# Intersect the K1 rank
(ab_k1_coords, ab_k1_a_handles, ab_k1_b_handles) = Intersect(a_k1_coords, a_k1_handles, b_k1_coords, b_k1_handles)
# Only retrieve the fibers that survive intersection
a_k1_payloads = HandlesToPayloads(a_k1, ab_k1_a_handles)
b_k1_payloads = HandlesToPayloads(b_k1, ab_k1_b_handles)
a_k0s = PayloadsToFiberHandles(a_k1, a_k1_payloads)
b_k0s = PayloadsToFiberHandles(b_k1, b_k1_payloads)


# k0 & operator:
a_k0_handless = Scan(a_k0s)
b_k0_handless = Scan(b_k0s)
a_k0_coordss = HandlesToCoords(a_k0s, a_k0_handless)
b_k0_coordss = HandlesToCoords(b_k0s, b_k0_handless)
# Intersect the K0 rank
(ab_k0_coordss, ab_k0_a_handless, ab_k0_b_handless) = Intersect(a_k0_coordss, a_k0_handless, b_k0_coordss, b_k0_handless)
# Only retrieve the values that survive intersection
a_k0_payloadss = HandlesToPayloads(a_k0s, ab_k0_a_handless)
b_k0_payloadss = HandlesToPayloads(b_k0s, ab_k0_b_handless)
a_valuess = PayloadsToValues(a_k0s, a_k0_payloadss)
b_valuess = PayloadsToValues(b_k0s, b_k0_payloadss)

# Compute result and reduce

# Original sequential code
#partial_productss = Compute(lambda a, b: a * b, a_valuess, b_valuess)

# BEGIN PARALLEL_FOR
NUM_PES = 4
dist_func = lambda n: n % NUM_PES 
k0_distribution_choicess = Compute(dist_func, ab_k0_coordss)
a_valuess_distributed = Distribute(4, k0_distribution_choicess, a_valuess)
b_valuess_distributed = Distribute(4, k0_distribution_choicess, b_valuess)

body_func = lambda a, b: a * b
resultss = []
for pe in range(NUM_PES):
  resultss.append(Compute(body_func, a_valuess_distributed[pe], b_valuess_distributed[pe], instance_name=str(pe)))
 
partial_productss = Collect(NUM_PES, k0_distribution_choicess, resultss)
# END PARALLEL FOR

partial_sums = Reduce(partial_productss)
z_root_new_value = Reduce(partial_sums)
z_root_handle = Stream0(0) # XXX Improve this
z_root_ack = UpdatePayloads(z_root, z_root_handle, z_root_new_value)


K1 = 2
K0 = 3
my_a_root = BasicIntermediateRankImplementation(1, 1)
my_a_k1 = BasicIntermediateRankImplementation(K1, K0)
my_a_k0 = [BasicFiberImplementation([1, 2, 3]), BasicFiberImplementation([2, 4, 6])]
my_b_root = BasicIntermediateRankImplementation(1, 1)
my_b_k1 = BasicIntermediateRankImplementation(K1, K0)
my_b_k0 = [BasicFiberImplementation([4, 5, 6]), BasicFiberImplementation([8, 10, 12])]
my_z_root = BasicFiberImplementation([0])

a.setImplementations("root", [my_a_root])
a.setImplementations("K1", [my_a_k1])
a.setImplementations("K0", my_a_k0)
b.setImplementations("root", [my_b_root])
b.setImplementations("K1", [my_b_k1])
b.setImplementations("K0", my_b_k0)
z.setImplementations("root", [my_z_root])

evaluate(z_root_ack, 0)

expected_val = 160

print("==========================")
print(f"Final K1-K0 result:")
print(my_z_root.vals)
assert(my_z_root.vals == [expected_val])
print("==========================")
