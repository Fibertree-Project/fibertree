import os

from fibertree import Tensor

print("--------------------------------------")
print("         Elementwise copy")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor(os.path.join(data_dir, "elementwise-a.yaml"))
z = Tensor(rank_ids=["M"])

a.print("A Tensor")
z.print("Z Tensor")

a_m = a.root()
z_m = z.root()

a_m.print("A Tensor - Rank M")
z_m.print("Z Fiber - Rank M")

print("Z < A Fiber")

for coord, (z_ref, a_val) in z_m << a_m:
    print("Processing: (%s, (%s, %s))" % (coord, z_ref, a_val))

    z_ref += a_val

z.print("\nZ Fiber - Rank M")

print("")
print("--------------------------------------")
print("")

