
import os

from fibertree import Tensor

#
# Merge-based matrix-vector multiply
#

print("--------------------------------------")
print("      Merge-based spMspV")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor.fromYAMLfile(os.path.join(data_dir, "spMspV-a-t.yaml"))
b = Tensor.fromYAMLfile(os.path.join(data_dir, "spMspV-b.yaml"))

z = Tensor(rank_ids=["M"])

a.print("A Tensor")
b.print("B Tensor")
z.print("Z Tensor")

a_k = a.getRoot()
b_k = b.getRoot()
z_m = z.getRoot()

a_k.print("A Tensor - Rank K")
b_k.print("B Tensor - Rank K")
z_m.print("Z Tensor - Rank M")

ab = a_k & b_k
ab_m = ab.swapRanks()

for m_coord, (z_ref, ab_k) in z_m << ab_m:
    for k_coord, (a_val, b_val) in ab_k:
        z_ref += a_val * b_val # reducing a scalar

z.print("\nZ Tensor")

print("")
print("--------------------------------------")
print("")
