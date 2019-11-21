from fibertree.tensor import Tensor
from fibertree.fiber import Fiber


a = Tensor(rank_ids=["M"], n=4)
b = Tensor(rank_ids=["M"], n=5)
z = None

a_m = a.root()
b_m = b.root()
z_m = Fiber(default=0)

a_m.print("A Fiber - M rank")
b_m.print("B Fiber - M rank")
z_m.print("Z Fiber - M rank")

print("Z < A Fiber")

for coord, (z_ref, (a_val, b_val)) in z_m << (a_m & b_m):
    print("Processing: (%s, (%s, (%s, %s)))"
          % (coord, z_ref, a_val, b_val))

    z_ref += a_val * b_val

print("")
a_m.print("A Fiber")
b_m.print("B Fiber")
z_m.print("Z Fiber")

