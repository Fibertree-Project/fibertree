Hierarchical Fiber-Tree-based Tensor Abstraction
==================================================

An emulator for the hierarchical fiber-tree abstraction for tensors

Install
=======

To install an editable copy in your home directory run:

```
  # Install the package
  % pip3 install --user -e .

  # To get widgets in the Jupyter notebooks
  jupyter nbextension enable --py widgetsnbextension
```

Run examples
============
Next you can run the included examples. For example::

```
  % cd examples/scripts
  % python3 dot-product.py
```

Other examples are in examples/scripts

Run Ipython notebooks
=====================

To run the Ipython notebooks:

```
  % cd examples/ipython
  % jupyter notebook .
```

Then open basic/fibertree.ipyn 


Run tests
=========

```
   % cd test
   % python3 -m unittest discover [-v]
```
