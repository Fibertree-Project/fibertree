Hierarchical Fibertree-based Tensor Abstraction
==================================================

An emulator for the hierarchical fibertree abstraction for
tensors. For a description of the concepts and some some designs
rendered in a simplified version of this system see sections 8.2 and
8.3 of the book "Efficient Processing of Deep Neural Networks" [1]. 

Install
=======

To install an editable copy in your home directory run the following
in the root directory of a clone of this repository:

```console
  % python3 -m pip install --user -e .

```
To install from the remote git repository, run the following:

```console
  % python3 -m pip install git+https://github.com/Fibertree-Project/fibertree
```

To install with Cython (`fibertree.core` in Cython, remaining files in Python):
```
 python setup.py build_ext --inplace
```


Explore Jupyter Notebooks
=========================

Clone [fibertree notebooks](https://github.com/Fibertree-Project/fibertree-notebooks) 
for some example fibertree-based algorithms in Jupypter notebooks.


Run commmand line examples
===========================

You can also run the included examples from the command line. For
example::

```console
  % cd ./examples/scripts/basic
  % python3 dot-product.py
```

Other examples are in **./examples/scripts/...**


Run tests
=========

```console
   % cd ./test
   % python3 -m unittest discover [-v]
```



References
==========

[1] "[Efficient Processing of Deep Neural Networks](http://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1530)",
Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer, Synthesis
Lectures on Computer Architecture, June 2020, 15:2, 1-341.
