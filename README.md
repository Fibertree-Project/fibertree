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


FAQ
===

Q: How do I fix font-related errors when displaying graphics?

A: On Ubuntu/Debian systems you can try installing fonts-freefon-ttf with:

```
   apt install fonts-freefont-ttf
```

   If you know where the fonts are on your system then you can set the
   environment variable FIBERTREE_FONT in Python code you can do this
   with something like:

```
   import os

   os.environ['FIBERTREE_FONT'] = 'Pillow/Tests/fonts/FreeMono.ttf'
```
