from fibertree import Fiber

f = Fiber([0, 1, 2, 10, 12, 31, 41], [ 0, 10, 20, 100, 120, 310, 410 ])

print("Original fiber\n")
f.print()

#
# Unform coordiate-based split
#
coords = 10
print("Uniform coordinate split (groups of %s coordinates)\n" % coords)

fibers = f.splitUniform(coords)

for s in fibers:
    s.print()

#
# Non-unform coordiate-based split
#
splits = [0, 12, 31]
print("NonUniform coordinate split (splits at %s)\n" % splits)

fibers = f.splitNonUniform(splits)

for s in fibers:
    s.print()

#
# Equal position-based split
#
size = 2
print("Equal position split (groups of %s)\n" % size)

fibers = f.splitEqual(size)

for s in fibers:
    s.print()


sizes = [1, 2, 4]
print("NonEqual position split (splits of sizes %s)\n" % sizes)

fibers = f.splitUnEqual(sizes)

for s in fibers:
    s.print()

