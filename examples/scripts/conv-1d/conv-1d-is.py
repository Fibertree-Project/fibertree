import os

from fibertree import Tensor

print("----------------------------------------")
print("    Convolution 1-D Input Stationary")
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

w_r.print("W Tensor - R rank - size=%s" % W)
i_h.print("I Tensor - H rank - size=%s" % I)
o_q.print("O Tensor - Q rank - size=%s" % I)

print("Convolution")

for h, (i_val) in i_h:
    print(f"Processing input: ({h}, ({i_val}))")
    for q, (o_q_ref, w_val) in o_q << w_r.project(lambda r: h-r, (0, Q)):
        print(f"  Processing output ({q}, ({o_q_ref}, {w_val})")
        o_q_ref += w_val * i_val

o.print("\nOutput Tensor")

print("")
print("--------------------------------------")
print("")

