import os

from fibertree import Tensor

print("--------------------------------------")
print("        Elementwise multiply")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor.fromYAMLfile(os.path.join(data_dir, "elementwise-a.yaml"))
b = Tensor.fromYAMLfile(os.path.join(data_dir, "elementwise-b.yaml"))
z = Tensor(rank_ids=["M"])

a.print("A Tensor")
b.print("B Tensor")
z.print("Z Tensor")

a_m = a.getRoot()
b_m = b.getRoot()
z_m = z.getRoot()

a_m.print("A Tensor - M rank")
b_m.print("B Tensor - M rank")
z_m.print("Z Tensor - M rank")

print("Z < A Fiber")

for m_coord, (z_ref, (a_val, b_val)) in z_m << (a_m & b_m):
    print(f"Processing: ({m_coord}, ({z_ref}, ({a_val}, {b_val})))")


    z_ref += a_val * b_val

z.print("\nZ Tensor")

print("")
print("--------------------------------------")
print("")

