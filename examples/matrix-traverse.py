from fibertree.tensor import Tensor


print("--------------------------------------")
print("           Matrix Traverse")
print("--------------------------------------")
print("")

a = Tensor(rank_ids = ["M", "K"], n=6)

a_m = a.root()

a_m.print("Matrix")

for m, (a_k) in a_m:
    print("(%s, %s)"% (m, a_k))
    for k, (a_val) in a_k:
        print("Processing: (%s, %s)"% (k, a_val))

print("")
print("--------------------------------------")
print("")

