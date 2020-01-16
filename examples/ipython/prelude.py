# Library imports

import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#
# Import tensor class
#
from fibertree import Payload, Fiber, Tensor, TensorImage

#
# Import display classes/utilities
#
from IPython.display import display # to display images

def displayTensor(t):
    display(TensorImage(t).im)


def displayGraph(am_s):
    gr = nx.DiGraph()

    for (s, am_d) in am_s:
        gr.add_node(s)
        for (d, _) in am_d:
            gr.add_edge(s, d)
   
    pos = nx.spring_layout(gr)
    nx.draw(gr, pos, node_size=500, with_labels=True)
    plt.show()
  
#
# Matplolib classes (not currently used)
#
from matplotlib.pyplot import imshow

#
# Import rc to configure animation for HTML5
#
from matplotlib import rc
rc('animation', html='html5')

#
# Helper for data directory
#
import os

data_dir = "../data"

def datafileName(filename):
    return os.path.join(data_dir, filename)
