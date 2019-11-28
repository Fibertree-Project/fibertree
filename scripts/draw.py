#!/usr/bin/python3

import os
import argparse

from fibertree import Tensor, TensorImage

data_dir = "../examples/data"
example_file = os.path.join(data_dir, "draw-a.yaml")

parser = argparse.ArgumentParser(description='Display a tensor')
parser.add_argument("tensorfile", nargs="?", default=example_file)
args = parser.parse_args()

filename = args.tensorfile

a = Tensor(filename)
a.print(filename)

i = TensorImage(a)
i.show()
