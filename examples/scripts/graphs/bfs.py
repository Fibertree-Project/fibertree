import os

from fibertree import Tensor
from fibertree import Payload

print("----------------------------------------")
print("          BFS graph traversal")
print("----------------------------------------")
print("")

data_dir = "../../data"

#a = Tensor.fromYAMLfile(os.path.join(data_dir, "graph-a.yaml"))

# Adjacency matrix
a = Tensor.fromUncompressed([ "S", "D"],
                            [ [ 0, 1, 1, 0, 0, 0 ],
                              [ 0, 0, 1, 1, 0, 0 ],
                              [ 0, 0, 0, 1, 1, 0 ],
                              [ 0, 0, 0, 0, 1, 1 ],
                              [ 1, 0, 0, 0, 0, 1 ],
                              [ 1, 1, 0, 0, 0, 0 ] ])

# Fringe (current and next)
f0 = Tensor.fromUncompressed([ "D" ], [ 1, 0, 0, 0, 0, 0 ])

# Distance
d = Tensor(rank_ids=[ "S" ])

# Get root fibers
a_s = a.getRoot()
f0_d = f0.getRoot()
d_d = d.getRoot()

print("BFS")

level = 1


while (f0_d.countValues() > 0):
    f0_d.print("\nFringe")

    f1 = Tensor(rank_ids=[ "D" ]) 
    f1_d = f1.getRoot()

    for s, (_, a_d) in f0_d & a_s:
        print(f"Processing source {s}")
        print(f"Neighbors:\n {a_d}")

#        print(f"\na_d:\n{a_d}")
#        print(f"\nd_d:\n{d_d})")

#        a_less_d = a_d - d_d
#        print(f"\na_less_d:\n{a_less_d})")

#        assignment1 = d_d << a_less_d
#        print(f"\nd_d << (a_d - d_d):\n{assignment1}")

#        assignment2 = f1_d << assignment1
#        print(f"\nf1_d << (d_d << (a_d - d_d)):\n{assignment2}")

        for d, (f1_ref, (d_ref, _)) in f1_d << (d_d << a_d):
            print(f"  Processing destination {d} = {d_ref}")

            if Payload.isEmpty(d_ref):
                print(f"Adding destination {d}")

                f1_ref += 1
                d_ref += level

    level += 1
    f0 = f1
    f0_d = f0.getRoot()

d_d.print("\nDistance Tensor")

print("")
print("--------------------------------------")
print("")

