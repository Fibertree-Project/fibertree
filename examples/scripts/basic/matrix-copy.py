import os

from fibertree import Tensor

print("--------------------------------------")
print("         Matrix copy")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor.fromYAMLfile(os.path.join(data_dir, "matrix-a.yaml"))
z = Tensor(rank_ids=["M", "K"])

a.print("A Tensor")
z.print("Z Tensor")

a_m = a.getRoot()
z_m = z.getRoot()

a_m.print("A Tensor - M rank")
z_m.print("Z Tensor - M rank")

for m, (z_k, a_k) in z_m << a_m:
    print(f"Processing: Coord: {m}")
    print(f"   z_k: {z_k}")
    print(f"   a_k: {a_k}")
          
    for k, (z_ref, a_val) in z_k << a_k:
        z_ref += a_val

print("")
a.print("A Tensor")
z.print("Z Tensor")

print("")
print("--------------------------------------")
print("")

