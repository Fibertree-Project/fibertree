from fibertree import Tensor, TensorImage


a = Tensor.fromUncompressed(root=2)
print(a)

i = TensorImage(a)
i.show()
