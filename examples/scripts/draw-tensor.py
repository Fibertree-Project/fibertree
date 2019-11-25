from fibertree import Tensor, TensorImage

filename = "../data/draw-a.yaml"

a = Tensor(filename)
a.print(filename)

i = TensorImage(a)
i.show()
