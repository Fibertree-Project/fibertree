from fibertree import Tensor

print("--------------------------------------")
print("         Matrix copy")
print("--------------------------------------")
print("")


a = Tensor("../data/matrix-a.yaml")
z = Tensor(rank_ids=["M", "K"])

a.print("A Tensor")
z.print("Z Tensor")

a_m = a.root()
z_m = z.root()

a_m.print("A Tensor - M rank")
z_m.print("Z Tensor - M rank")

for m, (z_k, a_k) in z_m << a_m:
    print("Processing: Coord: %s" % m)
    print("   z_k: %s" % z_k)
    print("   a_k: %s" % a_k)
          
    for k, (z_ref, a_val) in z_k << a_k:
        z_ref += a_val

print("")
a.print("A Tensor")
z.print("Z Tensor")

print("")
print("--------------------------------------")
print("")

