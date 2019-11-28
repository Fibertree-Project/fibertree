import os

from fibertree import Tensor, TensorImage

data_dir = "../../data"
filename = os.path.join(data_dir, "draw-a.yaml")

a = Tensor(filename)
a.print(filename)

i = TensorImage(a)
i.show()
