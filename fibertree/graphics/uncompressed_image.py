"""Uncompressed Image Module"""

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
module_logger = logging.getLogger('fibertree.graphics.uncompressed_image')


class UncompressedImage():
    """UncompressedImage

    This class is used to draw an uncompressed representation of a tensor

    Constructor
    -----------

    Parameters
    ----------

    object: tensor or fiber
        A tensor or fiber object to draw

    highlights: dictionary
        A dictionary of "workers" each with list of points to highlight

    extent: tuple
        Maximum row/col to use for image
    """

    def __init__(self, object, highlights={}, extent=(100, 200), row_map=None):
        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.uncompressed_image')

        #
        # Record paramters
        #
        # Note: We conditionally unwrap Payload objects
        #
        self.object = Payload.get(object)
        self.row_extent = extent[0]
        self.col_extent = extent[1]
        self.row_map = row_map

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
        # Draw the tensor
        #
        self._create_uncompressed()


    def _create_uncompressed(self):
        """Create uncompressed image

        Create an image of a tensor or fiber tree

        Notes
        ------

        The drawing is made in a coordinate space where the X
        and Y are the positions in the tensor.
        Translation to pixels happens in the draw_*() methods.

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
            # Handle a tensor
            #
            root = object.getRoot()
            self._color = object.getColor()
            #
            # Print tensor name
            #
            name = object.getName()
            if not name:
                name = "unknown"

            ranks = ", ".join([str(r) for r in object.getRankIds()])

            self._draw_label(0, 0, f"Tensor: {name}[{ranks}]")

        elif isinstance(object, Fiber):
            #
            # Handle a fiber
            #
            root = object
            self._color = "red"
        else:
            #
            # Handle a scalar
            #
            root = None
            self._color = "red"

        #
        # Process appropriately if root has 0 dimensions or more
        #
        if not Payload.contains(root, Fiber):
            # Draw a 0-D tensor, i.e., a value

            # TBD
            region_size = [1, 1]

        else:
            # Draw a non-0-D tensor or a fiber, i.e., the fiber tree
            region_size = self._traverse(root)

        #
        # Crop the image
        #
        if region_size[0] > self.row_extent or region_size[1] > self.col_extent:
            msg = f"Uncompressed image too large [ {region_size[0]}, {region_size[1]}"
            self.logger.info(msg)
            return

        right = 200+self._col2x(region_size[1])
        lower = 20+self._row2y(region_size[0])

        self.logger.debug(f"right: {region_size[1]}/{right}, lower: {region_size[0]}/{lower}")
        self.im = self.im.crop((0, 0, right, lower))


    def show(self):
        """Show the fibertree image"""

        self.im.show()


#
# Method to traverse (and draw) all the cells in the tensor
#
    def _traverse(self, fiber):
        """traverse"""

        #
        # Assume this is a rank-3 or less tensor
        #
        if not Payload.contains(fiber, Fiber):
            #
            # Draw a 0-D tensor, i.e., a value (NOT a fiber)
            #

            # TBD
            region_size = [1, 1]
        else:
            #
            # Recursively draw the fibers of a non-0-D tensor
            #
            shape = fiber.getShape(all_ranks=True)
            dimensions = len(shape)

            hl_manager = self.highlight_manager

            if dimensions == 4:
                region_size = self._traverse_hypercube(shape, fiber, highlight_manager=hl_manager)
            elif dimensions == 3:
                region_size = self._traverse_cube(shape, fiber, highlight_manager=hl_manager)
            elif dimensions == 2:
                region_size = self._traverse_matrix(shape, fiber, highlight_manager=hl_manager)
            elif dimensions == 1:
                region_size = self._traverse_vector(shape, fiber, highlight_manager=hl_manager)
            else:
                self.logger.info(f"Unsupported number of ranks for uncompressed image ({dimensions})")
                region_size = [1, 1]

        return region_size


    def _traverse_hypercube(self, shape, fiber, row_origin=1, col_origin=0, highlight_manager=None):
        """ traverse_hypercube - unimplemented """

        self.logger.debug("Display a hypercube")

        #
        # Print out the rank information (if available)
        #
        self._draw_label(row_origin, col_origin, "Rank: "+self._getId(fiber))
        self._draw_label(row_origin+1, col_origin, "|")
        self._draw_label(row_origin+2, col_origin, "V")

        row_cur = row_origin + 3
        row_max = row_origin + 3
        col_cur = col_origin

        #
        # Just show the nonEmpty cubes
        #
        for cube_c, cube_p in fiber:

            self._draw_label(row_cur, col_origin, f"{cube_c}")
            row_cur += 2
            row_max = row_cur

            highlight_manager_next = highlight_manager.addFiber(cube_c)

            self.logger.debug(f"Coord: {cube_c} - draw as [{row_cur}, {col_origin}]")

            rc_range = self._traverse_cube(shape[1:],
                                           cube_p,
                                           row_origin=row_cur,
                                           col_origin=col_origin,
                                           highlight_manager=highlight_manager_next)

            self.logger.debug(f"Coord: {cube_c} - rc_range: {rc_range}")

            # row_cur does not change
            row_cur = rc_range[0] + 2
            row_max = row_cur

            # col_cur does not change
            col_max = max(row_max, rc_range[1])

        return [row_max, col_max]


    def _traverse_cube(self, shape, fiber, row_origin=1, col_origin=0, highlight_manager=None):
        """ traverse_cube """

        self.logger.debug(f"Drawing cube at [{row_origin}, {col_origin}]")
        #
        # Print out the rank information (if available)
        #
        self._draw_label(row_origin, col_origin, "Rank: "+self._getId(fiber)+" ----->")

        row_cur = row_origin + 1
        row_max = row_origin + 1
        col_cur = col_origin
        col_max = col_origin

        #
        # Just show the nonEmpty matrices
        #
        for matrix_c, matrix_p in fiber:

            self._draw_label(row_origin, col_cur+5, f"{matrix_c}")

            highlight_manager_next = highlight_manager.addFiber(matrix_c)

            rc_range = self._traverse_matrix(shape[1:],
                                            matrix_p,
                                            row_origin=row_cur,
                                            col_origin=col_cur,
                                            highlight_manager=highlight_manager_next)

            # row_cur does not change
            row_max = max(row_max, rc_range[0])

            col_cur = rc_range[1] + 2
            col_max = col_cur

            if col_cur > self.col_extent: break

        return [row_max, col_max]



    def _traverse_matrix(self, shape, fiber, row_origin=1, col_origin=0, highlight_manager=None):
        """ traverse_matrix """

        #
        # Print out the rank information (if available)
        #
        label = "Rank: "+self._getId(fiber)
        self._draw_label(row_origin+2, col_origin, label)

        #
        # Set up variables to track rows and columns (note offset for rank label)
        #
        row_cur = row_origin
        row_max = row_cur

        col_cur = col_origin + (len(label)+2)//3
        col_max = col_cur

        #
        # Set up for loop
        #
        row_p = Fiber([], [])
        row_first = True

        #
        # For integer coordinates traverse all the coordinates in the shape
        # otherwise traverse all the non-empty coordinates
        #
        coords = range(shape[0])

        if isinstance(fiber, Fiber):
            if len(fiber) > 0 and not isinstance(fiber.coords[0], int):
                coords = fiber.coords


        for row_c in coords:

            if self.row_map:
                coord_label = str(self.row_map[row_c])
            else:
                coord_label = row_c

            if fiber is not None:
                row_p = fiber.getPayload(row_c)

            highlight_manager_next = highlight_manager.addFiber(row_c)

            rc_range = self._traverse_vector(shape[1:],
                                             row_p,
                                             row_origin=row_cur,
                                             col_origin=col_cur,
                                             highlight_manager=highlight_manager_next,
                                             rank_label=row_first,
                                             coord_label=coord_label)

            row_max = max(row_max, rc_range[0])
            row_cur = row_max
            row_first = False

            # col_cur does not change
            col_max = max(col_max, rc_range[1])

            if row_cur > self.row_extent: break


        return [row_max, col_max]


    def _traverse_vector(self,
                         shape,
                         fiber,
                         row_origin=1,
                         col_origin=0,
                         highlight_manager=None,
                         rank_label=True,
                         coord_label=None):

        #
        # Print out the rank information (if available)
        #
        # TBD: Align column more inteligently
        #
        if coord_label is not None:
            col_hack = 3
        else:
            col_hack = 0

        if rank_label:
            self._draw_label(row_origin, col_origin+col_hack, "Rank: "+self._getId(fiber))

            for c in range(fiber.getShape(all_ranks=False)):
                self._draw_label(row_origin+1, col_origin+col_hack+c, f"{c:^3}")

            rank_label_offset = 2
        else:
            rank_label_offset = 0

        #
        # Handle spans of empty rows
        #
        if len(fiber) != 0 or rank_label:
            #
            # On non-empty (or first) row reset empty row counter
            #
            self._empty_count = 0
        else:
            #
            # After first row, check for empty rows
            #
            self._empty_count += 1

            if self._empty_count == 2:
                self._draw_label(row_origin, col_origin+col_hack, "...")
                return [ row_origin+1, col_origin]

            if self._empty_count > 2:
                return [ row_origin, col_origin]

        #
        # Print out coordinate information (if available)
        #
        if coord_label is not None:
            try:
                label = f"{coord_label:>9}"
            except Exception:
                label = f"{str(coord_label):>9}"

            self._draw_label(row_origin+rank_label_offset, col_origin, label)
            coord_label_offset = col_hack
        else:
            coord_label_offset = 0


        #
        # Set up variables to track rows and columns
        #
        # Note: offsets for rank and coordinate labels
        #
        row_cur = row_origin + rank_label_offset
        row_max = row_origin + rank_label_offset

        col_cur = col_origin + coord_label_offset
        col_max = col_origin + coord_label_offset

        #
        # Determine if coordinates are integers
        #
        if len(fiber) > 0 and isinstance(fiber.coords[0], int):
            coord_is_int = True
        else:
            coord_is_int = False

        #
        # Process each coordinate in the shape
        #
        for coord in range(shape[0]):
            #
            # Get highlighting information from highlight manager
            #
            color_coord = highlight_manager.getColorCoord(coord)
            color_subtensor = highlight_manager.getColorSubtensor()
            color_coord_or_subtensor = color_coord | color_subtensor

            if isinstance(fiber, Fiber):
                #
                # For printing a non-empty row
                #
                if coord_is_int:
                    payload = fiber.getPayload(coord)
                else:
                    #
                    # Just show non-integer coordinates in order
                    #
                    try:
                        payload = fiber.payloads[coord]
                    except:
                        payload = 0

                assert not isinstance(payload, Fiber)
            else:
                #
                # For printing a empty row
                #
                payload = 0

            row_count = self._draw_value(row_cur, col_cur, payload, color_coord_or_subtensor)

            # row_cur does not change
            row_max = max(row_max, row_cur+row_count)

            col_cur += 1
            col_max = col_cur

            if col_max > self.col_extent: break


        return [row_max, col_max]

#
# Utility methods
#

    def _getId(self, fiber):
        """ _getId - get fiber's rank id """

        if fiber.getOwner() is None:
            return ""

        return str(fiber.getOwner().getId())

#
# Image methods
#
    def _image_setup(self):

        # Constrain image size (overage matches crop above)

        x_pixels = self._col2x(self.col_extent+1) + 200 # was 8192
        y_pixels = self._row2y(self.row_extent+1) + 20 # was 1024

        # Create an image at least this tall (in pixels)
        self.max_y = 100


        # Do image related setup
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
#          - row
#          - column
#
    def _draw_label(self, row, column, label):
        """draw_label"""

        x1 = self._col2x(column) + 20
        y1 = self._row2y(row) - 10

        # Hack: drawing text twice looks better in PIL
        self.draw.text((x1+10,y1+10), label, font=self.fnt, fill="black")
        self.draw.text((x1+10,y1+10), label, font=self.fnt, fill="black")



    def _draw_value(self, row, column, value, highlight=[]):
        """draw_value"""

        #
        # Check if we're outside the box
        #
        if row >= self.row_extent or column >= self.col_extent:
            if row == self.row_extent or column == self.col_extent:
                self._draw_label(row, column, "...")
                return 2

            return 0


        if isinstance(value, Payload):
            value = value.value

        if not isinstance(value, tuple):
            value = ( value, )

        row_count = len(value)

        font_y = 30

        x1 = self._col2x(column) + 20
        y1 = self._row2y(row) - 10

        x2 = x1 + 40
        y2 = y1 + row_count*(font_y+10)
        
        if y2 > self.max_y:
            self.max_y = y2


        if len(highlight) == 0:
            fill_color = self._color if value != (0, ) else 0
            self.draw.rectangle(((x1,y1), (x2,y2)), fill_color, 1)
        else:
            step = (y2-y1) // len(highlight)
            y1c = y1
            for worker in highlight:
                y2c = y1c + step
                fill_color = self.worker_color[worker]
                self.draw.rectangle(((x1,y1c), (x2,y2c)), fill_color, 1)
                y1c = y2c

        for i, v in enumerate(value):
            if isinstance(v, Payload):
                v = v.value

            x_text = x1+15
            y_text = y1+10+(i*font_y)
            if (isinstance(v, int)):
                abs_v = abs(v)

                if v < 0:
                    x_text = x_text - 7
                if abs_v >= 10:
                    x_text = x_text - 7
                if abs_v >= 100:
                    x_text = x_text - 7
                if abs_v >= 1000:
                    x_text = x_text - 7
            elif (isinstance(v, float)):
                v = round(v, 2)


            # Hack: drawing text twice looks better in PIL
            self.draw.text((x_text, y_text),
                            str(v),
                            font=self.fnt,
                            fill="white")
            self.draw.text((x_text, y_text),
                            str(v),
                            font=self.fnt,
                            fill="white")

        return row_count
#
#
# Methods to convert positions specified in col/row space into pixels
#
    def _col2x(self, col):
        return 200 + 40*col
            
    def _row2y(self, row):
        return 40 + 40*row


if __name__ == "__main__":
                         
    print("a - multiple highlights")
    a = Tensor.fromYAMLfile("../../examples/data/sparse-matrix-a.yaml")
    a.setColor("blue")
    i = UncompressedImage(a, highlights={"PE": [(0, 1), (1, 2), (3,)]})
    i.show()

    #
    print("a - single highlights")
    i = UncompressedImage(a, {"PE": [(1, 2)]})
    i.show()

    #
    print("b")
    b = Tensor.fromUncompressed(["X"], [1, 2, 0, 0, 4])
    i = UncompressedImage(b, {"PE": [(1,), (4,)]})
    i.show()

    #
    print("c")
    a_root = a.getRoot()
    c = Tensor.fromFiber(["X", "Y", "Z"], Fiber([0, 1, 2], [a_root, Fiber([],[]), a_root]))
    i = UncompressedImage(c)
    i.show()

    #
    print("d")
    d = c.getRoot()
    print("Original")
    i = UncompressedImage(d)
    i.show()

    #
#    print("e")
#    d_flattened = d.flattenRanks()
#    print("Flattened")
#    i = UncompressedImage(d_flattened)
#    i.show()
