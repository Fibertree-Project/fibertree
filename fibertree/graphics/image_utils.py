from PIL import Image, ImageDraw, ImageFont

class ImageUtils():
    """ImageUtils

    A utility class for supporting graphics for multiple drawing
    classes. A number of global attributes are class variables of this
    class

    """

    #
    # Map worker names to colors
    #
    hl_colors = [0xdaa520,   # worker 0 d (goldenrod)
                 0x977316,   # worker 1 a
                 0xe4b849,   # worker 2 f
                 0xc4941d,   # worker 3 c
                 0xea1f33,   # worker 4 e
                 0xae8319,   # worker 5 b
                 0xe8c15f]   # worker 6 g

    #
    # Next color to allocate
    #
    hl_next = 0
    
    #
    # Map of worker names to colors
    #
    hl_map = {}


    def __init__(self):
        """__init__ """

        pass
        

    @staticmethod
    def getColor(worker):
        """ getColor() """

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
        """ resetColors """

        ImageUtils.hl_next = 0
        ImageUtils.hl_map = {}
