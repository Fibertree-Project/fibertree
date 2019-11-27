from fibertree import Fiber

a = Fiber( [0, 1, 10, 20 ], [ 1, 2, 11, 21 ])

ap = a.project(lambda c: c + 1)

a.print("a\n")
ap.print("ap\n")


