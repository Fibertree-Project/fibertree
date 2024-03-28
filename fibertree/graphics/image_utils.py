"""Image Utilities Module"""

import logging
import os
import webcolors

from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

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
    def getFont(font_name=None, font_size=16):
        """Get a font for use in images.

        Get a standard font for various image classes to use. First
        looks for a file as specified by environment variable
        "FIBERTREE_FONT", then at a well-known location.

        To set the environment variable in Python try the following:

        import os
        os.environ['FIBERTREE_FONT'] = 'Pillow/Tests/fonts/FreeMono.ttf'


        TBD: Make more robust for use on different systems

        """

        #
        # Find the path to the data file within the package
        #
        if font_name is None:
            font_name = "FreeMono"

        data_file_path = f"fonts/{font_name}.ttf"

        #
        # Try getting location from environment
        #
        font_file = os.getenv('FIBERTREE_FONT')

        if font_file is not None:
            try:
                font = ImageFont.truetype(font_file, font_size)
                return font
            except Exception as e:
                print(f"Could not find font file: {font_file}")
                raise e

        #
        # Get ffile from installed package
        #
        try:
            import importlib.resources as importlib_resources

            ref = importlib_resources.files("fibertree") / data_file_path
            with importlib_resources.as_file(ref) as font_file:
                try:
                    font = ImageFont.truetype(str(font_file), font_size)
                    return font
                except Exception as e:
                    print(f"Could not find font file: {font_file}")
                    raise e
        except (AttributeError, ModuleNotFoundError):
            cur_dir = os.path.abspath(os.path.dirname(__file__))
            font_file = os.path.join(cur_dir, data_file_path)
            try:
                font = ImageFont.truetype(font_file, font_size)
                return font
            except Exception as e:
                print(f"Could not find font file: {font_file}")
                raise e


    @staticmethod
    @lru_cache
    def pick_text_color(bg_color):
        """
        Selects a text color of black or white to best go with a given background color.

        Args:
          bg_color: The background color in string or RGB format.

        Returns:
          The text color, either 'black' or 'white'.
        """

        #
        # Conditionally convert `bg_color` to RBG
        #
        if isinstance(bg_color, str):
            if bg_color[0] != '#':
                try:
                    bg_color = webcolors.name_to_rgb(bg_color)
                except Exception as err:
                    return 'black'
            else:
                bg_color = ImageUtils.hex2rgb(bg_color)

        #
        # Calculate the brightness of the background color.
        #
        brightness = (bg_color[0] * 0.299 + bg_color[1] * 0.587 + bg_color[2] * 0.114) / 255

        #
        # If the brightness is less than 0.5, return 'white'.
        #
        if brightness < 0.5:
            return 'white'
        #
        # Otherwise, return 'black'.
        #
        else:
            return 'black'


    @staticmethod
    def hex2rgb(hex_value):

        # Remove the '#' symbol if present
        hex_value = hex_value.lstrip('#')

        # Convert the hexadecimal string to integer
        hex_int = int(hex_value, 16)

        # Extract the RGB components
        red = (hex_int >> 16) & 255
        green = (hex_int >> 8) & 255
        blue = hex_int & 255

        # Return the RGB components as an array
        return [red, green, blue]
