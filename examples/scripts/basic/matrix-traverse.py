import os

from fibertree import Tensor

print("--------------------------------------")
print("           Matrix Traverse")
print("--------------------------------------")
print("")

data_dir = "../../data"

a = Tensor.fromYAMLfile(os.path.join(data_dir, "matrix-a.yaml"))

a.print("Matrix")

a_m = a.getRoot()

a_m.print("Matrix - M Fiber")

for m, (a_k) in a_m:
    print(f"({m}, {a_k})")
    for k, (a_val) in a_k:
        print(f"Processing: ({k}, {a_val})")

print("")
print("--------------------------------------")
print("")

