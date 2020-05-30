""" IPython prelude code """
#
# Argument parsing imports
# Note: to guide animation display since people often do "Run All Cells"
#
import argparse

#
# System imports
#
import os
import string
import random

#
# Import display classes/utilities
#
from IPython.display import display # to display images
from IPython.display import Image
from IPython.display import HTML
from IPython.display import Javascript

#
# Math imports
#
import math

try:
    import numpy as np
except ImportError:
    print("Library numpy not available")

try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow
    from matplotlib import rc
except ImportError:
    print("Library matplotlib not available")

have_networkx = True
try:
    import networkx as nx
except ImportError:
    have_networkx = False

#
# Import tensor class
#
from fibertree import Payload, Fiber, Tensor, TensorImage, TensorCanvas

#
# Try to import ipywidgets
#
have_ipywidgets = True
try:
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
except ImportError:
    have_ipywidgets = False

#
# Use rc to configure animation for HTML5
#
rc('animation', html='html5')


class FibertreeDisplay():
    """ FibertreeDisplay """

    def __init__(self, have_ipywidgets=False):
        """ __init__ """

        self.have_ipywidgets = have_ipywidgets

        self.style = 'tree'
        self.enable_animations = False

        self.setupWidgets()

    #
    # Display control settings
    #
    def setStyle(self, style='tree', sync=True):
        """ setStyle """

        if style not in ['tree', 'uncompressed', 'tree+uncompressed']:
            print("Unsuppored display style")
            return

        self.style = style

        if sync:
            self.syncWidgets()

    def showAnimations(self, enable_animations, sync=True):
        """ showAnimations """

        self.enable_animations = enable_animations

        if sync:
            self.syncWidgets()


    #
    # Display actions
    #
    def displayTensor(self, tensor, highlights=[], **kwargs):
        """ displayTensor """

        im = TensorImage(tensor, style=self.style, highlights=highlights, **kwargs).im

        display(im)


    def createCanvas(self, *tensors, **kwargs):
        """ createCanvas """

        return TensorCanvas(*tensors, style=self.style, **kwargs)


    def addFrame(self, canvas, *points):
        """ addFrame """

        if not self.enable_animations:
            return None

        return canvas.addFrame(*points)


    def displayCanvas(self, canvas, filename=None, width="100%", loop=True, autoplay=True, controls=True, center=False):
        """ displayCanvas """

        AnimationDisabledError = "Note: Canvas animation has been disabled - showing final frame"

        if not self.enable_animations:
            #
            # Just create a frame from the last state and display it
            #
            canvas.addFrame()
            im = canvas.getLastFrame(AnimationDisabledError)
            display(im)
            return

        if filename is None:
            filename = self.random_string(10)

        canvas.saveMovie(filename + ".mp4")

        # Append random string to URL to prevent browser caching
        randomstring=self.random_string(10)
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

    def random_string(self, length):
        return ''.join(random.choice(string.ascii_letters) for m in range(length))


    def displayGraph(self, am_s):
        """ displayGraph """

        if not have_networkx:
            print("Library networkx not available")
            return

        gr = nx.DiGraph()

        for (s, am_d) in am_s:
            gr.add_node(s)
            for (d, _) in am_d:
                gr.add_edge(s, d)

        pos = nx.spring_layout(gr)
        nx.draw(gr, pos, node_size=500, with_labels=True)
        plt.show()

    #
    # Widget control
    #
    def setupWidgets(self):
        """ setupWidgets """

        if have_ipywidgets:
            self.w = interactive(self.updateWidgets,
                                 style=['tree', 'uncompressed', 'tree+uncompressed'],
                                 animation=['enabled', 'disabled'])

            display(self.w)
        else:
            print("Warning: ipywidgets not available - set attributes manually by typing:")
            print("")
            print("FTD.setStyle('uncompressed')           # Show tensor as a uncompressed")
            print("FTD.setStyle('tree')                   # Show tensor as a fiber tree")
            print("FTD.setStyle('tree+uncompressed')      # Show tensor in both styles")
            print("")
            print("FTD.showAnimations(True)               # Turn on animations")
            print("FTD.showAnimations(False)              # Turn off animations")
            print("")
            

    def updateWidgets(self, style='tree', animation='disabled'):
        """ setup """

        #
        # Set attributes (but do not recurse back and sync widgets)
        #
        self.setStyle(style=style, sync=False)
        self.showAnimations(animation == 'enabled', sync=False)

    def syncWidgets(self):
        """ syncWidgets """

        style = self.style

        if self.enable_animations:
            animation = 'enabled'
        else:
            animation = 'disabled'

        if self.have_ipywidgets:
            self.w.children[0].value = style
            self.w.children[1].value = animation
        else:
            print(f"Style: {style}")
            print(f"Animation:  {animation}")
            print("")

#
# Functions for a "run_all" button
#
def run_all_below(ev):
    """ run_all_below """

    display(Javascript('IPython.notebook.select_next()'))
    display(Javascript('IPython.notebook.execute_cells_below()'))


def createRunallButton():
    """ createRunallButton """

    button = widgets.Button(description="Run all cells below")
    button.on_click(run_all_below)
    display(button)


#
# Functions for use in the IPython notebooks
#

#
# Convenience functions that just call the class methods
# on the FTD object created below
#
def displayTensor(tensor, highlights=[], **kwargs):
    """ displayTensor(<tensor|fiber>, hightlights=[ <point>...] ) """

    FTD.displayTensor(tensor, highlights=highlights, **kwargs)


def displayGraph(am_s):
    """ displayGraph(am_s) """

    FTD.displayGraph(am_s)


def createCanvas(*tensors, **kwargs):
    """ createCanvas """

    return FTD.createCanvas(*tensors, **kwargs)


def addFrame(canvas, *points):
    """ addFrame """

    return FTD.addFrame(canvas, *points)


def displayCanvas(*args, **kwargs):
    """ displayCanvas """

    FTD.displayCanvas(*args, **kwargs)

#
# Helper function for locating the data directory
#
data_dir = "../../data"

def datafileName(filename):
    return os.path.join(data_dir, filename)


#
# Parse the arguments (deprecated)
#
parser = argparse.ArgumentParser()
parser.add_argument('--show-animations', dest='EnableAnimations', action='store_true')
parser.add_argument('--no-show-animations', dest='EnableAnimations', action='store_false')
parser.set_defaults(EnableAnimations=True)
args = parser.parse_args()

#
# Instantiate the FiberTree Display class
#
FTD = FibertreeDisplay(have_ipywidgets)

FTD.showAnimations(args.EnableAnimations)

#
# If possible create a runall button
#
if have_ipywidgets:
    createRunallButton()
