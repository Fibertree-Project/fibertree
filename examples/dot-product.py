from fibertree.tensor import Tensor
from fibertree.fiber  import Fiber

#
# To do a dot-product we need a "row" for an output.
# So we represent the vectors as 2-D tensors
#

a = Tensor(rank_ids=["M","K"], n=1)
b = Tensor(rank_ids=["M","K"], n=2)

z = Fiber(default=0)

a_m = a.root()
b_m = b.root()

a_m.print("A Tensor - Rank M")
b_m.print("B Tensor - Rank M")
z.print("Z Tensor - Rank M")

for m_coord, (z_ref, (a_k, b_k)) in z << (a_m & b_m):
    for k_coord, (a_val, b_val) in a_k & b_k:
        print("Processing: [%s -> ( %s, (%s, %s)]"
              % (k_coord, z_ref, a_val, b_val))

        z_ref += a_val * b_val

z.print("\nZ Tensor")
