
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

a = Tensor(os.path.join(data_dir, "spMspV-a.yaml"))
b = Tensor(os.path.join(data_dir, "spMspV-b.yaml"))

z = Tensor(rank_ids=["M"])

a.print("A Tensor")
b.print("B Tensor")
z.print("Z Tensor")

a_m = a.root()
b_k = b.root()
z_m = z.root()

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
