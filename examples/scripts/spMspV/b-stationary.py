
import os

from fibertree import Tensor

#
# B-stationary matrix-vector multiply
# i.e., outer-product style
# or A-stationary column-major style

print("--------------------------------------")
print("      B-stationary spMspV")
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

for k_coord, (a_m, b_val) in (a_k & b_k):
    for m_coord, (z_ref, a_val) in (z_m << a_m):
        z_ref += a_val * b_val # reducing a vector

z.print("\nZ Tensor")

print("")
print("--------------------------------------")
print("")
