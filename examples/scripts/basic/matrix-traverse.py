import os

from fibertree import Tensor

print("--------------------------------------")
print("           Matrix Traverse")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor(os.path.join(data_dir, "matrix-a.yaml"))

a.print("Matrix")

a_m = a.root()

a_m.print("Matrix - M Fiber")

for m, (a_k) in a_m:
    print("(%s, %s)" % (m, a_k))
    for k, (a_val) in a_k:
        print("Processing: (%s, %s)"% (k, a_val))

print("")
print("--------------------------------------")
print("")

