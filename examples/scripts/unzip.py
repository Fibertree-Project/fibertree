from fibertree import Fiber

ab = Fiber( [0, 1, 10, 20 ], [ (0, 1), (1, 2), (10, 11), (20, 21) ])

(a, b) = ab.unzip()

ab.print("ab\n")
a.print("a\n")
b.print("b\n")


