import os

from fibertree import Tensor

print("----------------------------------------")
print("    Convolution 1-D Weight Stationary")
print("----------------------------------------")
print("")

data_dir = "../../data"

w = Tensor.fromYAMLfile(os.path.join(data_dir, "conv-weights-a.yaml"))
i = Tensor.fromYAMLfile(os.path.join(data_dir, "conv-activations-a.yaml"))
o = Tensor(rank_ids=["Q"])

w.print("W Tensor")
i.print("I Tensor")
o.print("O Tensor")

w_r = w.getRoot()
i_h = i.getRoot()
o_q = o.getRoot()

W = w_r.maxCoord() + 1
I = i_h.maxCoord() + 1
Q = I - W + 1

w_r.print(f"W Tensor - R rank - size={W}")
i_h.print(f"I Tensor - H rank - size={I}")
o_q.print(f"O Tensor - Q rank - size={Q}")

print("Convolution")

for r, (w_val) in w_r:
    print(f"Processing weight: ({r}, ({w_val}))")
    for q, (o_q_ref, i_val) in o_q << i_h.project(lambda h: h-r, (0, Q)):
        print(f"  Processing output ({q}, ({o_q_ref}, {i_val})")
        o_q_ref += w_val * i_val

o.print("\nOutput Tensor")

print("")
print("--------------------------------------")
print("")

