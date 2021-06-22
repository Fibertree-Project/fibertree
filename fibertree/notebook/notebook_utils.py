""" Ipython Notebook Utilities """

#
# Import standard libraries
#
import logging
from pathlib import Path

#
# Import display classes/utilities
#
from IPython.display import display # to display images
from IPython.display import Image
from IPython.display import HTML
from IPython.display import Javascript

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#
# Import fibertree libraries
#
from fibertree import Tensor


class NotebookUtils():

    def __init__(self):

        #
        # Debugging variables
        #
        logging.basicConfig(format='%(asctime)s %(message)s')

        self.logger = logging.getLogger("fibertree.notebook")

        self.types = { "Debug":    logging.DEBUG,
                       "Info":     logging.INFO,
                       "Warning":  logging.WARNING,
                       "Error":    logging.ERROR,
                       "Critical": logging.CRITICAL}

        self.modules = { 'Notebook':            "fibertree.notebook",
                         "Tensor":              "fibertree.core.tensor",
                         "Rank":                "fibertree.core.rank",
                         "Fiber":               "fibertree.core.fiber",
                         "Coord_Payload":       "fibertree.core.coord_payload",
                         "Payload":             "fibertree.core.payload",
                         "Tensor_canvas":       "fibertree.graphics.tensor_canvas",
                         "Movie_canvas":        "fibertree.graphics.movie_canvas",
                         "Spacetime_canvas":    "fibertree.graphics.spacetime_canvas",
                         "Tensor_image":        "fibertree.graphics.tensor_image",
                         "Tree_image":          "fibertree.graphics.tree_image",
                         "Uncompressed_image":  "fibertree.graphics.uncompressed_image",
                         "Highlights":          "fibertree.graphics.highlights",
                         "Image_utils":         "fibertree.graphics.image_utils"}


        self.levels = {}

    #
    # Functions for a "run_all" button
    #
    @staticmethod
    def run_all_below(ev):
        """ run_all_below """

        display(Javascript('IPython.notebook.select_next()'))
        display(Javascript('IPython.notebook.execute_cells_below()'))


    def createRunallButton(self):
        """ createRunallButton """

        button = widgets.Button(description="Run all cells below")
        button.on_click(self.run_all_below)
        display(button)


    #
    # Logging functions
    #
    def getLogger(self):
        """ getLogger """

        return self.logger


    def showLogging(self, **kwargs):
        """ showLogging """

        for m_name, m_level in kwargs.items():
            self.levels[m_name] = m_level

        controls = {}

        style = {'description_width': 'initial'}

        for m_name in self.modules.keys():
            controls[m_name] = widgets.Dropdown(options=[*self.types],
                                                value=self.levels.get(m_name,'Warning'),
                                                description=m_name,
                                                style=style,
                                                disabled=False)


        print("")
        print("Debugging level")
        display(interactive(self._set_debug, **controls))
        print("")

        return


    def _set_debug(self, **kwargs):
        """ _set_debug """

        for m_name, m_level in kwargs.items():
            print(m_name, m_level, self.modules[m_name], self.types[m_level])
            logger = logging.getLogger(self.modules[m_name])
            logger.setLevel(self.types[m_level])

        return


#
# Functions for use in the IPython notebooks
#

#
# Function for a Boolean enable dropdown
#
# TBD: Deprecate use of this method
#

enable = {}

def createEnableControl(name, choices=None):
    """ createEnableControl

    Create a widget with a dropdown box for setting
    the variable "enable[name]" with "choices".

    """

    def set_enable(**kwargs):

        global enable

        for key, value in kwargs.items():
            enable[key] = value

    if choices is None:
        choices = [False, True]

    kwargs = {name: choices}

    w = interactive(set_enable,
                    **kwargs)

    display(w)


#
# Helper function for locating the data directory
#
# TBD: Deprecate use of this method
#
data_dir = Path("../../data")

def datafileName(filename):

    return data_dir / filename


#
# Deprecated canvas functions
#
def addFrame(canvas, *args, **kwargs):
    """ addFrame """

    msg = "The method addFrame() is deprecated. Use canvas.AddFrame()"
    warnings.warn(msg, FutureWarning, stacklevel=3)

    canvas.addFrame(*args, **kwargs)


def addActivity(canvas, *args, **kwargs):
    """ addActivity """

    msg = "The method addActivity() is deprecated. Use canvas.AddActivity()"
    warnings.warn(msg, FutureWarning, stacklevel=3)

    canvas.addActivity(*args, **kwargs)

