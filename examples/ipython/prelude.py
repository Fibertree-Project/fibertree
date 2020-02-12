# Library imports

import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#
# Import tensor class
#
from fibertree import Payload, Fiber, Tensor, TensorImage, TensorCanvas

#
# Import display classes/utilities
#
from IPython.display import display # to display images
from IPython.display import Image
from IPython.display import HTML
import string
import random

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
# Arguments to guide animation display since people often do "Run All Cells"
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--show-animations', dest='DisableAnimations', action='store_false')
parser.add_argument('--no-show-animations', dest='DisableAnimations', action='store_true')
parser.set_defaults(DisableAnimations=False)
args = parser.parse_args()

AnimationDisabledError = """Note: Canvas animation has been disabled by --no-show-animations
        (Typically this option is set in the first cell)
        Showing final frame
"""

def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for m in range(length))

def displayCanvas(canvas, filename=None, width="100%", loop=True, autoplay=True, controls=True, center=False):
    if args.DisableAnimations:
        im = canvas.getLastFrame(AnimationDisabledError)
        display(im)
        return
    if filename is None:
        filename=random_string(10)

    canvas.saveMovie(filename + ".mp4")

    # Append random string to URL to prevent browser caching
    randomstring=random_string(10)
    final_width = "" if width is None else " width=\"{0}\"".format(width)
    final_loop = "" if not loop else " loop"
    final_autoplay = "" if not autoplay else " autoplay"
    final_controls = "" if not controls else " controls"
    final_center = "" if not center else " style=\"display:block; margin: 0 auto;\""
    html = """
        <video{}{}{}{}{}>
            <source src="{}.mp4?t={}" type="video/mp4">
        </video>
      """
    display(HTML(html.format(final_width, final_loop, final_autoplay, final_controls, final_center, filename, randomstring)))
    #
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

print("Prelude loaded OK")
if args.DisableAnimations:
    print("(Animation display disabled)")
