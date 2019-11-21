from fibertree.tensor import Tensor
from fibertree.fiber import Fiber


a = Tensor(rank_ids=["M"], n=3)
z = None

a_m = a.root()
z_m = Fiber([6, 17], [20, 22])

a_m.print("A Tensor - Rank M")
z_m.print("Z Fiber - Rank M")

print("Z < A Fiber")

for coord, (z_ref, a_val) in z_m << a_m:
    print("(%s, (%s, %s))" % (coord, z_ref, a_val))

    z_ref += a_val

a_m.print("A Fiber - Rank M")
print("")
z_m.print("Z Fiber - Rank M")
print("")

