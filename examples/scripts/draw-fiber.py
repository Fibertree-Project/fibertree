from fibertree import Fiber, Tensor, TensorImage

a = Fiber([ 0, 2, 8], [ 5, 6, 7 ])

a.print("Fiber")
i = TensorImage(a)
i.show()
