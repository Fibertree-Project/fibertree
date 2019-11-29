from fibertree import Fiber, Tensor, TensorImage

x = [ [ [ 1, 2, 8, 20, 0, 0, 11 ],
        [ 1, 0, 0, 11, 0, 0, 33 ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 1, 1, 8, 12, 0, 0, 44 ],
        [ 1, 3, 0, 13, 0, 0, 42 ],
        [ 0, 0, 4, 14, 0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ] ],

      [ [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ] ],

      [ [ 1, 2, 8, 20, 0, 0, 11 ],
        [ 1, 0, 0, 11, 0, 0, 33 ],
        [ 0, 0, 0, 0,  0, 0, 0  ],
        [ 1, 1, 8, 12, 0, 0, 44 ],
        [ 1, 3, 0, 13, 0, 0, 42 ],
        [ 0, 0, 4, 14, 0, 0, 0  ],
        [ 0, 0, 0, 0,  0, 0, 0  ] ] ]

            

  

f = Fiber.fromUncompressed(x)
f.dump("/tmp/tensor-3d.yaml")
f.print("Fiber from uncompressed")

t1 = Tensor.fromFiber(["X", "Y", "Z"], f)
t1.print("Tensor from fiber")

t2 = Tensor.fromUncompressed(["X", "Y", "Z"], x)
t2.print("Tensor from uncompressed")
