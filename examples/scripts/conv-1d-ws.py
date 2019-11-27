from fibertree import Tensor

print("----------------------------------------")
print("    Convolution 1-D Weight Stationary")
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

for r, (w_val) in w_r:
    print("Processing weight: (%s, (%s))" % (r, w_val))
    for q, (o_q_ref, i_val) in o_q << i_h.project(lambda h: h-r, (0, Q)):
        print("  Processing output (%s, (%s, %s)" % (q, o_q_ref, i_val))
        o_q_ref += w_val * i_val

o.print("\nOutput Tensor")

print("")
print("--------------------------------------")
print("")

