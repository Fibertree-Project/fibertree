Hierarchical Fiber-Tree-based Tensor Abstraction
==================================================

An emulator for the hierarchical fiber-tree abstraction for
tensors. For a description of the concepts and some some designs
rendered in a simplified version of this system see sections 8.2 and
8.3 of the book "Efficient Processing of Deep Neural Networks" [1].

Install
=======

To install an editable copy in your home directory run:

```console
  % pip3 install --user -e .
```

Run examples
============
Next you can run the included examples. For example::

```console
  % cd ./examples/scripts/basic
  % python3 dot-product.py
```

Other examples are in **./examples/scripts/...**

Run Ipython notebooks
=====================

To run the Ipython notebooks:


```console
  % cd ./examples/ipython
  % jupyter notebook .
```

Then open **basic/fibertree.ipynb**

See FAQ below for addressing some problems.

Run tests
=========

```console
   % cd ./test
   % python3 -m unittest discover [-v]
```


FAQ
===

Q: How do I fix font-related errors when displaying graphics?

A: On Ubuntu/Debian systems you can try installing fonts-freefon-ttf with:

```console
   % sudo apt install fonts-freefont-ttf
```

   If you know where the fonts are on your system then you can set the
   environment variable FIBERTREE_FONT in Python code you can do this
   with something like:

```python
   import os

   os.environ['FIBERTREE_FONT'] = 'Pillow/Tests/fonts/FreeMono.ttf'
```

Q: How can I get the Jupyter widgets to work?

A1: For classic Jupyter, try the following command:

```console
  % jupyter nbextension enable --py widgetsnbextension
```

A2: To get widgets to work in Jupyter lab, try the following:

```console
  % curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.6/install.sh | bash
  # restart bash
  % nvm install node
  % jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


References
==========

[1] "[Efficient Processing of Deep Neural Networks](http://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1530)",
Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer, Synthesis
Lectures on Computer Architecture, June 2020, 15:2, 1-341.
