"""Image Utilities Module"""

import logging
import os

from PIL import Image, ImageDraw, ImageFont

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.image_utils')


class ImageUtils():
    """ImageUtils

    A utility class for supporting graphics for multiple drawing
    classes. A number of global attributes are class variables of this
    class.

    """

    hl_colors = ["goldenrod",
                 "#efcf62",   # worker 0 - yellow
                 "#85b5c9",   # worker 2 - aqua
                 "#dd7820",   # worker 1 - orange
                 "#90bf89",   # worker x - light green
                 "#daa520",   # worker 3 - goldenrod
                 "#91a9f1",   # worker 4 - light blue
                 "#ea1f33",   # worker 4 e
                 "#ae8319",   # worker 5 b
                 "#e8c15f"]   # worker 6 g
    """A pre-defined set of colors for highlighting workers."""
    #
    # Next color to allocate
    #
    hl_next = 0
    """The index of the next color to assign as a highlight."""
    
    #
    # Map of worker names to colors
    #
    hl_map = {}
    """A hash map of worker names (spacestamps) to colors."""


    def __init__(self):
        """__init__ """

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.image_utils')

        
    @staticmethod
    def setColor(worker, color):
        """Set color for a worker.

        Parameters
        ----------

        worker: hashable value
            Name of a worker (spacestamp)

        color: Pillow color
            Color to associate with `worker`

        """

        hl_map = ImageUtils.hl_map

        if worker in hl_map:
            print(f"WARNING: {worker} already has a color - OVERWRITING!")

        hl_map[worker] = color


    @staticmethod
    def getColor(worker):
        """ Get color associated with a worker.

        If no color is currently assigned to `worker`, then assign one
        round-robin from `hl_colors`.

        Parameters
        ----------

        worker: hashable value
            Name of a worker (spacestamp)


        """

        hl_map = ImageUtils.hl_map

        if worker in hl_map:
            return hl_map[worker]

        #
        # Allocate next color
        #
        hl_next = ImageUtils.hl_next
        hl_colors = ImageUtils.hl_colors

        color = hl_colors[hl_next]
        hl_map[worker] = color

        ImageUtils.hl_next = (hl_next + 1) % len(hl_colors)

        return color


    @staticmethod
    def resetColors():
        """Clear all worker colors."""

        ImageUtils.hl_next = 0
        ImageUtils.hl_map = {}


    @staticmethod
    def getFont():
        """Get a font for use in images.

        Get a standard font for various image classes to use. First
        looks for a file as specified by environment variable
        "FIBERTREE_FONT", then at a well-known location.

        To set the environment variable in Python try the following:

        import os
        os.environ['FIBERTREE_FONT'] = 'Pillow/Tests/fonts/FreeMono.ttf'


        TBD: Make more robust for use on different systems

        """

        font_file = os.getenv('FIBERTREE_FONT')

        if font_file is None:
            font_file = 'Pillow/Tests/fonts/FreeMono.ttf'

        try:
            font = ImageFont.truetype(font_file, 20)
            return font
        except Exception as e:
            print(f"Could not find font file: {font_file}")
            raise e
