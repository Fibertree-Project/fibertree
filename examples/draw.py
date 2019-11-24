#!/usr/bin/python3

import argparse
from fibertree import Tensor, TensorImage

parser = argparse.ArgumentParser(description='Display a tensor')
parser.add_argument("tensorfile", nargs="?", default="./data/draw-a.yaml")
args = parser.parse_args()

filename = args.tensorfile

a = Tensor(filename)
a.print(filename)

i = TensorImage(a)
i.show()
