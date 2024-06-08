"""Fibertree Display Module"""

#
# Import standard libraries
#
import os
import string
import random
import tempfile

from pathlib import Path

#
# Import display classes/utilities
#
from IPython.display import display # to display images
from IPython.display import Image
from IPython.display import HTML
from IPython.display import Javascript
from IPython.display import Video

from base64 import b64encode

from .movie_player import MoviePlayer
from .slideshow_player import SlideshowPlayer

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
# Import matplotlib
#
try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import imshow
    from matplotlib import rc
except ImportError:
    print("Library matplotlib not available")

#
# Try to import networkx
#
have_networkx = True
try:
    import networkx as nx
except ImportError:
    have_networkx = False

#
# Import fibertree libraries
#
from fibertree import TensorImage, TensorCanvas


class TensorDisplay():
    """ FibertreeDisplay """

    def __init__(self, style=None, animation=None, have_ipywidgets=False, create_dialog=True):
        """ __init__ """

        self.have_ipywidgets = have_ipywidgets

        self.style = 'tree'
        self.animation = 'none'

        if create_dialog:
            self.setupWidgets()

            if style is not None:
                self.setStyle(style)

            if animation is not None:
                self.setAnimation(animation)


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

    def setAnimation(self, animation='none', sync=True):
        """ setAnimation """

        self.animation = animation

        if sync:
            self.syncWidgets()


    #
    # Display actions
    #
    def displayTensor(self, tensor, highlights=[], style=None, **kwargs):
        """ displayTensor """

        if style is None:
            style = self.style

        im = TensorImage(tensor,
                         highlights=highlights,
                         style=style,
                         **kwargs)

        display(im.im)


    def createCanvas(self, *tensors, style=None, animation=None, **kwargs):
        """ createCanvas """

        if style is None:
            style = self.style

        if animation is None:
            animation = self.animation

        return TensorCanvas(*tensors,
                            style=style,
                            animation=animation,
                            **kwargs)


    def displayCanvas(self,
                      canvas,
                      layout=None,
                      animation=None,
                      filename=None,
                      width="100%",
                      loop=True,
                      autoplay=True,
                      controls=True,
                      center=False):

        """ displayCanvas """

        if canvas is None:
            return None

        #
        # Only can do a last minute swap between "slideshow" and "movie"
        #
        if animation == "slideshow" and self.animation != "movie":
            animation = self.animation

        elif animation == "movie" and self.animation != "slidehow":
            animation = self.animation
        else:
            animation = self.animation

        if animation == 'none':
            #
            # Just create a frame from the last state and display it
            #
            AnimationDisabledError = "Note: Canvas animation has been disabled - showing final frame"

            im = canvas.getLastFrame(AnimationDisabledError)
            display(im)
            return

        if animation == 'spacetime':
            #
            # Get the spacetime diagrams
            #

            print("Spacetime")

            for image in canvas.getLastFrame():
                display(image)

            return

        if animation == 'slideshow':

            player = SlideshowPlayer(canvas,
                                     layout=layout)

            player.display()

            return

        if animation == 'movie':

            player = MoviePlayer(canvas,
                                 filename=filename,
                                 layout=layout)

            player.display(width,
                           loop,
                           autoplay,
                           controls)


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
                                 animation=['none', 'movie', 'slideshow', 'spacetime'])

            display(self.w)
        else:
            print("Warning: ipywidgets not available - set attributes manually by typing:")
            print("")
            print("FTD.setStyle('uncompressed')           # Show tensor as a uncompressed")
            print("FTD.setStyle('tree')                   # Show tensor as a fiber tree")
            print("FTD.setStyle('tree+uncompressed')      # Show tensor in both styles")
            print("")
            print("FTD.setAnimation('none')               # Turn off animations")
            print("FTD.setAnimation('movie')              # Turn on movie animation")
            print("FTD.setAnimation('spacetime')          # Turn on spacetime animation")
            print("")
            

    def updateWidgets(self, style='tree', animation='none'):
        """ setup """

        #
        # Set attributes (but do not recurse back and sync widgets)
        #
        self.setStyle(style=style, sync=False)
        self.setAnimation(animation=animation, sync=False)

    def syncWidgets(self):
        """ syncWidgets """

        style = self.style
        animation = self.animation

        if self.have_ipywidgets:
            self.w.children[0].value = style
            self.w.children[1].value = animation
        else:
            print(f"Style: {style}")
            print(f"Animation:  {animation}")
            print("")

