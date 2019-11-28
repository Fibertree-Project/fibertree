import os

from fibertree import Fiber, Tensor

print("---------------")
print("YAML I/O Tests")
print("---------------")

data_dir = "../../data"
tmp_dir = "/tmp"

# Read in a Tensor

draw_a_file = os.path.join(data_dir, "draw-a.yaml")
t1 = Tensor(yamlfile=draw_a_file)

# Dump the Tensor to /tmp"

draw_a_file_tmp = os.path.join(tmp_dir, "draw-a.yaml")
t1.dump(draw_a_file_tmp)

# Read in the Tensor from /tmp"

t2 = Tensor(yamlfile=draw_a_file_tmp)

print("Tensor read/write test: %s" % (t1.root() == t2.root()))

# Read in a Fiber

draw_fiber_a_file = os.path.join(data_dir, "draw-fiber-a.yaml")
f1 = Fiber(yamlfile=draw_fiber_a_file)

print("Fiber read test: %s" % (t1.root() == f1))

# Dump the Fiber to /tmp

draw_fiber_a_file_tmp = os.path.join(tmp_dir, "draw-fiber-a.yaml")
f1.dump(draw_fiber_a_file_tmp)

# Read in the Fiber from /tmp

f2 = Fiber(yamlfile=draw_fiber_a_file_tmp)

print("Fiber read/write test: %s" % (f1 == f2))

