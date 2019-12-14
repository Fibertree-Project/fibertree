import os

from fibertree import Fiber, Tensor

print("---------------")
print("YAML I/O Tests")
print("---------------")

data_dir = "../../data"
tmp_dir = "/tmp"

# Read in a Tensor

draw_a_file = os.path.join(data_dir, "draw-a.yaml")
t1 = Tensor.fromYAMLfile(draw_a_file)

# Dump the Tensor to /tmp"

draw_a_file_tmp = os.path.join(tmp_dir, "draw-a.yaml")
t1.dump(draw_a_file_tmp)

# Read in the Tensor from /tmp"

t2 = Tensor.fromYAMLfile(draw_a_file_tmp)

print(f"Tensor read/write test: {(t1.getRoot() == t2.getRoot())}")

# Read in a Fiber

draw_fiber_a_file = os.path.join(data_dir, "draw-fiber-a.yaml")
f1 = Fiber.fromYAMLfile(draw_fiber_a_file)

print(f"Fiber read test: {(t1.getRoot() == f1)}")

# Dump the Fiber to /tmp

draw_fiber_a_file_tmp = os.path.join(tmp_dir, "draw-fiber-a.yaml")
f1.dump(draw_fiber_a_file_tmp)

# Read in the Fiber from /tmp

f2 = Fiber.fromYAMLfile(draw_fiber_a_file_tmp)

print(f"Fiber read/write test: {(f1 == f2)}")

