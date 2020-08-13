
import os

from fibertree import Tensor

#
# C-stationary matrix-vector multiply
# i.e., output-stationary or inner-product style
# i.e., or A-stationary row major

print("--------------------------------------")
print("      C-stationary spMspV")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor.fromYAMLfile(os.path.join(data_dir, "spMspV-a.yaml"))
b = Tensor.fromYAMLfile(os.path.join(data_dir, "spMspV-b.yaml"))

z = Tensor(rank_ids=["M"])

a.print("A Tensor")
b.print("B Tensor")
z.print("Z Tensor")

a_m = a.getRoot()
b_k = b.getRoot()
z_m = z.getRoot()

a_m.print("A Tensor - Rank M")
b_k.print("B Tensor - Rank K")
z_m.print("Z Tensor - Rank M")

for m_coord, (z_ref, a_k) in (z_m << a_m):
    for k_coord, (a_val, b_val) in (a_k & b_k):
        z_ref += a_val * b_val # reducing a scalar

z.print("\nZ Tensor")

print("")
print("--------------------------------------")
print("")
