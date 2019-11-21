from fibertree.tensor import Tensor
from fibertree.fiber import Fiber

print("--------------------------------------")
print("         Elementwise copy")
print("--------------------------------------")
print("")


a = Tensor(rank_ids=["M"], n=3)
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

