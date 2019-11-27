from fibertree import Tensor

print("----------------------------------------")
print("    Convolution 1-D Input Stationary")
print("----------------------------------------")
print("")


w = Tensor("../data/conv-weights-a.yaml")
i = Tensor("../data/conv-activations-a.yaml")
o = Tensor(rank_ids=["Q"])

W = w.values()
I = i.values()
Q = I - 2

w.print("W Tensor - size=%s" % W)
i.print("I Tensor - size=%s" % I)
o.print("O Tensor - size=%s" % Q)

w_r = w.root()
i_h = i.root()
o_q = o.root()

w_r.print("W Tensor - R rank")
i_h.print("I Tensor - H rank")
o_q.print("O Tensor - Q rank")

print("Convolution")

for h, (i_val) in i_h:
    print("Processing input: (%s, (%s))" % (h, i_val))
    for q, (o_q_ref, w_val) in o_q << w_r.project(lambda r: h-r, (0, Q)):
        print("  Processing output (%s, (%s, %s)" % (q, o_q_ref, w_val))
        o_q_ref += w_val * i_val

o.print("\nOutput Tensor")

print("")
print("--------------------------------------")
print("")

