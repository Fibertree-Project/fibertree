"""Tree Image Module"""

import logging

from PIL import Image, ImageDraw

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from fibertree import ImageUtils
from fibertree import HighlightManager

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.tree_image')


class TreeImage():
    """TreeImage

    Class to create a fibertree image of a tensor

    Parameters
    ----------
    object: tensor or fiber
        A tensor or fiber object to draw

    highlights: dictionary
        A dictionary of "workers" each with list of points to highlight

    extent: tuple
        Maximum row/col to use for image

    """

    def __init__(self, object, highlights={}, extent=(30, 200)):
        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.tree_image')

        #
        # Record parameters
        #
        # Note: We conditionally unwrap Payload objects
        #
        self.object = Payload.get(object)
        self.row_extent = extent[0]
        self.col_extent = extent[1]

        level = self.object.getDepth()-1
        self.highlight_manager = HighlightManager(highlights, level=level)

        #
        # Cache worker colors
        #
        worker_color = {}

        for n, worker in enumerate(highlights.keys()):
            worker_color[worker] = ImageUtils.getColor(worker)

        self.worker_color = worker_color

        #
        # Create the tree image
        #
        self._create_tree()


    def _create_tree(self):
        """create_tree

        Create an image of a tensor or fiber tree

        Notes
        ------

        The drawing is made in a coordinate space where the X
        dimension is measured in the number of non-empty fiber
        coordinates being displayed and the Y dimension is measured in
        layers of the tree. Translation to pixels happens in the
        draw_*() methods.

        """

        object = self.object

        #
        # Create the objects for the image
        #
        self._image_setup()

        #
        # Display either the root of a tensor or a raw fiber
        #
        if isinstance(object, Tensor):
            #
            # Displaying a tensor
            #
            root = object.getRoot()
            #
            # Get tensor's name
            #
            name = object.getName()
            #
            # Get tensor's color
            #
            self._color = object.getColor()
            #
            # Create rank_id string
            #
            # Note: if rank_id is a list, convert to a string
            #
            ranks = ", ".join([str(r) for r in object.getRankIds()])

            if name:
                self._draw_rank(0, f"Tensor: {name}[{ranks}]")
            else:
                self._draw_rank(0, f"File: {object.yamlfile}")
        elif isinstance(object, Fiber):
            #
            # Displaying a fiber
            #
            root = object
            self._color = "red"
        else:
            #
            # Displaying nothing?
            #
            root = None
            self._color = "red"

        #
        # Process appropriately if root has 0 dimensions or more
        #
        if not Payload.contains(root, Fiber):
            #
            # Draw a 0-D tensor, i.e., a value
            #
            hlm = self.highlight_manager
            hl = hlm.getColorCoord(0)

            self._draw_coord(0, 0, "R")
            self._draw_line(0, 1/2, 1, 1/2)
            self._draw_value(1,
                             0,
                             Payload.get(root),
                             hl)
            region_end = 1
        else:
            #
            # Draw a non-0-D tensor or a fiber, i.e., the fiber tree
            #
            region_end = self._traverse(root,
                                       highlight_manager=self.highlight_manager)

        #
        # Crop the image
        #
        right = 200+self._offset2x(region_end)
        lower = 20+self.max_y

        self.im = self.im.crop((0, 0, right, lower))



    def show(self):
        """Show the fibertree image"""

        self.im.show()


#
# Method to traverse (and draw) all the levels of the tree
#
    def _traverse(self,
                 fiber,
                 level=0,
                 offset=0,
                 highlight_manager=None):
        """traverse"""


        #
        # Check if this is level0, which may just be a payload
        #
        if level == 0:
            region_start = 0

            if not Payload.contains(fiber, Fiber):
                #
                # Draw a 0-D tensor, i.e., a value (NOT a fiber)
                #
                self._draw_coord(0, 0, "R")
                self._draw_line(0, 1/2, 1, 1/2)
                self._draw_value(1, 0, Payload.get(fiber))
                region_end = 1
            else:
                #
                # Recursively traverse and draw the fibers of a non-0-D tensor
                #
                region_end = self._traverse(fiber,
                                            level=1,
                                            offset=offset,
                                            highlight_manager=highlight_manager)

                region_size = region_end - region_start
                #
                # Draw root of tree
                #
                fiber_size = 1
                fiber_start = region_start + (region_size - fiber_size)/2
                self._draw_coord(0, fiber_start, "R")
                self._draw_line(0, region_size/2, 1, region_size/2)

            return region_end

        #
        # Process the fibers of the tree (level > 0)
        #

        #
        # Print out the rank information (if available)
        #
        if offset == 0 and not fiber.getOwner() is None:
            self._draw_rank(level, "Rank: %s " % fiber.getOwner().getId())

        #
        # Initialize drawing region information
        #
        region_start = offset
        region_end = offset

        #
        # Figure out space of region below this fiber
        #
        targets = []
        coordinate_start = region_start

        #
        # Traverse the fiber at this level
        #
        for n, (c, p) in enumerate(fiber):
            #
            # TBD: Truncate fibers with too many elements
            #
            # if n > 10: break

            if Payload.contains(p, Fiber):
                #
                # Configure highlights for this fiber
                #
                next_highlight_manager = highlight_manager.addFiber(c)

                #
                # Draw the object below this coordinate (in "c")
                #
                region_end = self._traverse(Payload.get(p),
                                            level=level+1,
                                            offset=region_end,
                                            highlight_manager=next_highlight_manager)

            else:
                region_end += 1

            #
            # Record (in "targets") the middle of the object below
            # this coordinate to draw a line to it later, and
            # calculate where the next object starts ("coordinate_start")
            #
            targets.append(coordinate_start+(region_end-coordinate_start)/2)
            coordinate_start = region_end

        #
        # If the fiber was empty we still occupy a single space
        #
        if len(fiber) == 0:
            region_end += 1

        region_size = region_end - region_start

        #
        # Set up the highlighting for this level
        #
        highlight_subtensor = highlight_manager.highlight_subtensor

        #
        # Display fiber for this level
        #
        fiber_size = len(fiber)
        fiber_start = region_start + (region_size - fiber_size)/2

        self._draw_fiber(level,
                        fiber_start,
                        fiber_start+fiber_size,
                        highlight_subtensor)

        pos = fiber_start

        for c, p in fiber:
            #
            # Gets sets of workers to be colored
            #
            color_coord = highlight_manager.getColorCoord(c)
            color_subtensor = highlight_manager.getColorSubtensor()
            color_coord_or_subtensor = color_coord | color_subtensor

            #
            # Draw the coordinates, lines and maybe values
            #
            self._draw_coord(level, pos, c, color_coord_or_subtensor)

            if len(color_coord - color_subtensor):
                self._draw_intra_line(level,
                                     fiber_start + fiber_size / 2,
                                     pos+0.5,
                                     True)

            #
            # Draw the line if the next level will actually draw something.
            #
            if not Payload.contains(p, Fiber) or len(p.coords) > 0:
                self._draw_line(level,
                               pos+0.5,
                               level+1,
                               targets.pop(0),
                               len(color_coord_or_subtensor) > 0)
            else:
                #
                # Nothing to connect a line to so pop target
                #
                targets.pop(0)


            if not Payload.contains(p, Fiber):
                #
                # How could this not be the leaf ---
                # "and rest_of_highlighting == []"
                #
                self._draw_value(level+1,
                                pos,
                                Payload.get(p),
                                color_coord_or_subtensor)

            pos += 1

        return region_end

#
# Image methods
#
    def _image_setup(self):

        # Constrain image size (overage matches crop above)

        #
        # Size used to be (8192, 1024)
        #
        x_pixels = self._offset2x(self.col_extent+1) + 200
        y_pixels = self._level2y(self.row_extent+1) + 20

        #
        # Create an image at least this tall (in pixels)
        #
        self.max_y = 100

        #
        # Do image related setup
        #
        self.im = Image.new("RGB", (x_pixels, y_pixels), "wheat")
        self.fnt = ImageUtils.getFont()
        self.draw = ImageDraw.Draw(self.im)

#
#
#
    def _anti_alias(self, fill_color):
        r, g, b = fill_color
        return (r//4, g//4, b//4)


#
# Methods to draw objects on the drawing canvas
#
# Note: Input arguments place the objects at a position specified by:
#          - level: layer in the tree (Y)
#          - offset: number of drawn fiber coordinates (X)
#
    def _draw_rank(self, level, rank):
        """draw_rank"""

        x1 = 0
        y1 = self._level2y(level)

        # Hack: drawing text twice looks better in PIL
        self.draw.text((x1+10, y1+10), rank, font=self.fnt, fill="black")
        self.draw.text((x1+10, y1+10), rank, font=self.fnt, fill="black")


    def _draw_fiber(self, level, start_offset, end_offset, highlight=False):
        """draw_fiber"""

        height = 60
        gap = 5

        x1 = self._offset2x(start_offset) + gap
        y1 = self._level2y(level) - 10
        x2 = self._offset2x(end_offset) - gap
        y2 = y1 + height
        fill_color = (128,128,128) if not highlight else (233,198,109)

        self.draw.ellipse(((x1, y1), (x2, y2)), fill_color, (0, 0, 0))


    def _draw_coord(self, level, offset, coord, highlight=[]):
        """draw_coord"""

        x1 = self._offset2x(offset) + 20
        y1 = self._level2y(level)
        x2 = x1 + 40
        y2 = y1 + 40

        color = "goldenrod" if len(highlight) > 0 else "saddlebrown"

        x_text = x1+15

        if coord != "R" and isinstance(coord, int):
            if int(coord) >= 10:
                x_text = x_text - 7
            if int(coord) >= 100:
                x_text = x_text - 7
            if int(coord) >= 1000:
                x_text = x_text - 7

        if coord != "R":
            self.draw.ellipse(((x1,y1), (x2,y2)), color, 1)
        else:
            self._draw_diamond(x1, y1, x2, y2, "black")

        #
        # Hack: drawing text twice looks better in PIL
        #
        for n in range(2):
            self.draw.text((x_text, y1+10),
                           str(coord),
                           font=self.fnt,
                           fill="white")


    def _draw_value(self, level, offset, value, highlight=[]):
        """draw_value"""

        if isinstance(value, Payload):
            value = value.value

        if not isinstance(value, tuple):
            value = ( value, )

        value = self._flatten_value(value)

        font_y = 30

        #
        # Calculate location of rectangle around value
        #
        x1 = self._offset2x(offset) + 20
        y1 = self._level2y(level) - 10

        x2 = x1 + 40
        y2 = y1 + 10 + len(value)*(font_y)

        if y2 > self.max_y:
            self.max_y = y2

        #
        # Draw rectangle (with or without highlights)
        #
        if len(highlight) == 0:
            fill_color = self._color
            self.draw.rectangle(((x1, y1), (x2, y2)), fill_color, 1)
        else:
            step = (y2-y1) // len(highlight)
            y1c = y1
            for worker in highlight:
                y2c = y1c + step
                fill_color = self.worker_color[worker]
                self.draw.rectangle(((x1, y1c), (x2, y2c)), fill_color, 1)
                y1c = y2c

        #
        # Draw vertical stack of values from tuple
        #
        for i, v in enumerate(value):
            if isinstance(v, Payload):
                v = v.value

            #
            # Calculate location of text
            #
            x_text = x1+15
            y_text = y1+10+(i*font_y)

            if (isinstance(v, int)):
                if v >= 10:
                    x_text = x_text - 7
                if v >= 100:
                    x_text = x_text - 7
                if v >= 1000:
                    x_text = x_text - 7
            elif (isinstance(v, float)):
                v = round(v, 2)

            #
            # Hack: drawing text twice looks better in PIL
            #
            for n in range(2):
                self.draw.text((x_text, y_text),
                               str(v),
                               font=self.fnt,
                               fill="white")


    def _flatten_value(self, value, first=True):

        if isinstance(value, Payload):
            value = value.value

        if isinstance(value, tuple):
            result = []

            if not first:
                result.append("(")

            for i in value:
                result.extend(self._flatten_value(i, first=False))

            if not first:
                result.append(")")
        else:
            result = [ value ]

        return result


    def _draw_diamond(self, x1, y1, x2, y2, fill_color):
        mid_x = x1+(x2-x1)/2
        mid_y = y1+(y2-y1)/2

        self.draw.polygon([(mid_x, y1), (x1, mid_y), (mid_x, y2), (x2, mid_y)], fill_color, 1)


    def _draw_line(self, level1, offset1, level2, offset2, highlight=False):

        # Bottom of source is 40 below level2y result (see draw_coord)
        # Top of target is 10 above level2y results (see draw_fiber)

        x1 = self._offset2x(offset1)
        y1 = self._level2y(level1) + 40
        x2 = self._offset2x(offset2)
        y2 = self._level2y(level2) - 10

        fill_color = "goldenrod" if highlight else "black"

        self.draw.line([(x1, y1), (x2, y2)], width=3, fill=fill_color)


    def _draw_intra_line(self, level, fiber_offset, coord_offset, highlight=False):

        # Bottom of source is 10 above level2y results (see draw_line)
        # Top of target is level2y result (see draw_coord)

        x1 = self._offset2x(fiber_offset)
        y1 = self._level2y(level) - 10
        x2 = self._offset2x(coord_offset)
        y2 = self._level2y(level)

        fill_color = "goldenrod" if highlight else "black"

        self.draw.line([(x1, y1), (x2, y2)], width=3, fill=fill_color)

#
#
# Methods to convert positions specified in offset/level space into pixels
#
    def _offset2x(self, offset):
        return 200 + 80*offset


    def _level2y(self, level):
        return 40 + 80*level


if __name__ == "__main__":

    a = Tensor.fromYAMLfile("../../examples/data/draw-a.yaml")
    a.print()
    i = TreeImage(a)
    i.show()
