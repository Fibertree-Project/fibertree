Hierarchical Fiber-Tree-based Tensor Abstraction
==================================================

This directory contains a library for manipulating the fiber-tree abstraction
for tensor operations. Following is a rough grammar to try to illustrate the
use of the fiber-tree operators.


Major activities
================

Following are the expressions for some of the major activities on fiber trees

```

statement        :== <fiber> = <fiber expression>
                   | <payload> = <fiber name>.getPayload(<coordinate>)
                   | for <fiber element> in <fiber expression>: <statement>...


fiber expression ::= ( <fiber expression> )
                   | <fiber expression> <operator> <fiber expression>
                   | <fiber>


operator         ::= "<<" | "&" | "|" | "-"


fiber element    ::= (<coordinate>, <payload>)


coordinate       ::= <integer>


payload          ::= <integer value>
                   | <floating point value>
                   | <Fiber>
                   | ( <payload>, <payload>)

```


Naming conventions
==================

Some naming conventions used in the examples and Jupyter notebooks

```
for <rank name> ( <payload name> ) in <fiber expression>:


<payload name>    ::= <tensor name>_<rank name>         # for non-leaf rank
                    | <tensor name>_val                 # for an input at a leaf rank
                    | <tensor name>_ref                 # for an output at a leaf rank
                    | ( <payload name> , <payload name> )


<rank name>       ::= <letter> | <letter><digit>


```


Examples for some simple operators
===========================================

Assume the following fibers:

- "a_x" - a 1-D input fiber with rank "x"
- "b_x" - a 1-D input fiber with rank "x"
- "c_x" - a 1-D input fiber with rank "x"
- "d_x" - a 2-D input fiber with ranks "x" and "y"
- "z_x" - a 1-D output fiber with rank "x"


Following are some examples using the above naming conventions

```python

# Intersection (2-way)

for x, ( a_val, b_val ) in a_x & b_x:
   ....


# Intersection (3-way)

for x, ( a_val, (b_val, c_val) ) in a_x & (b_x & c_x):
   ....


# Intersection (2-D)

for x, ( a_val, d_y ) in a_x & d_x:
   for y, (d_val) in d_y:
        ...

# Union

for x, (mask, a_val, b_val ) in a | b:   # mask :== "AB" | "A" | "B"
   ....


# Difference

for x, ( a_val ) in a - b:
    ....


# Assignment

for x, (z_ref, a_val) in z << a:
    ....


```
