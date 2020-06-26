from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

class UncompressedImage():
    """UncompressedImage"""


    def __init__(self, object, highlights={}, extent=(30, 200), row_map=None):
        """__init__

        Parameters
        ----------

        object: tensor or fiber
        A tensor or fiber object to draw

        highlights: dictionary
        A dictionary of "workers" each with list of points to highlight

        extent: tuple
        Maximum row/col to use for image

    """
        #
        # Record paramters
        #
        # Note: We conditionally unwrap Payload objects
        #
        self.object = Payload.get(object)
        self.highlights = highlights
        self.row_extent = extent[0]
        self.col_extent = extent[1]
        self.row_map = row_map

        #
        # Map worker names to colors (copied from tree_image)
        #
        hl_colors = [0xdaa520,   # worker 0 d (goldenrod)
                     0x977316,   # worker 1 a
                     0xe4b849,   # worker 2 f
                     0xc4941d,   # worker 3 c
                     0xea1f33,   # worker 4 e
                     0xae8319,   # worker 5 b
                     0xe8c15f]   # worker 6 g

        worker_color = {}

        for n, worker in enumerate(highlights.keys()):
            worker_color[worker] = hl_colors[n % len(hl_colors)]

        self.worker_color = worker_color

        #
        # Draw the tensor
        #
        self.create_uncompressed()


    def create_uncompressed(self):
        """create_uncompressed: Create an image of a tensor or fiber tree

        Notes
        ------

        The drawing is made in a coordinate space where the X
        and Y are the positions in the tensor.
        Translation to pixels happens in the draw_*() methods.

        """

        object = self.object
        highlights = self.highlights

        #
        # Create the objects for the image
        #
        self.image_setup()

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

            self.draw_label(0, 0, f"Tensor: {name}[{ranks}]")

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
            region_size = self.traverse(root)

        #
        # Crop the image
        #
        self.im = self.im.crop((0,
                                0,
                                self.col2x(region_size[1])+200,
                                20+self.row2y(region_size[0])))


    def show(self):
        self.im.show()


#
# Method to traverse (and draw) all the cells in the tensor
#
    def traverse(self, fiber):
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


            if len(shape) == 3:
                region_size = self.traverse_cube(shape, fiber, highlights=self.highlights)
            elif len(shape) == 2:
                region_size = self.traverse_matrix(shape, fiber, highlights=self.highlights)
            elif len(shape) == 1:
                region_size = self.traverse_vector(shape, fiber, highlights=self.highlights)
            else:
                region_size = [1, 1]

        return region_size


    def traverse_cube(self, shape, fiber, row_origin=1, col_origin=0, highlights={}, highlight_subtensor={}):
        """ traverse_cube - unimplemented """

        #
        # Print out the rank information (if available)
        #
        self.draw_label(row_origin, col_origin, "Rank: "+self._getId(fiber)+" ----->")

        row_cur = row_origin + 1
        row_max = row_origin + 1
        col_cur = col_origin

        #
        # Just show the nonEmpty matrices
        #
        for matrix_c, matrix_p in fiber:

            self.draw_label(row_origin, col_cur+5, f"{matrix_c}")

            highlight_next, highlight_subtensor_next =  self._compute_next_highlights(matrix_c, highlights, highlight_subtensor)

            rc_range = self.traverse_matrix(shape[1:],
                                            matrix_p,
                                            row_origin=row_cur,
                                            col_origin=col_cur,
                                            highlights=highlight_next,
                                            highlight_subtensor=highlight_subtensor_next)

            # row_cur does not change
            row_max = max(row_max, rc_range[0])

            col_cur = rc_range[1] + 2
            col_max = col_cur

            if col_cur > self.col_extent: break


        return [row_max, col_max]



    def traverse_matrix(self, shape, fiber, row_origin=1, col_origin=0, highlights={}, highlight_subtensor={}):
        """ traverse_matrix """

        #
        # Print out the rank information (if available)
        #
        label = "Rank: "+self._getId(fiber)
        self.draw_label(row_origin+2, col_origin, label)

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
        # For non-integer coordinates traverse just them
        # otherwise traverse all the coordinates in the shape
        #
        if isinstance(fiber, Fiber) and not isinstance(fiber.coords[0], int):
            coords = fiber.coords
        else:
            coords = range(shape[0])

        for row_c in coords:

            if self.row_map:
                coord_label = str(self.row_map[row_c])
            else:
                coord_label = row_c

            if fiber is not None:
                row_p = fiber.getPayload(row_c)

            highlight_next, highlight_subtensor_next = self._compute_next_highlights(row_c, highlights, highlight_subtensor)

            rc_range = self.traverse_vector(shape[1:],
                                             row_p,
                                             row_origin=row_cur,
                                             col_origin=col_cur,
                                             highlights=highlight_next,
                                             highlight_subtensor=highlight_subtensor_next,
                                             rank_label=row_first,
                                             coord_label=coord_label)

            row_max = max(row_max, rc_range[0])
            row_cur = row_max
            row_first = False

            # col_cur does not change
            col_max = max(col_max, rc_range[1])

            if row_cur > self.row_extent: break


        return [row_max, col_max]


    def traverse_vector(self,
                        shape,
                        fiber,
                        row_origin=1,
                        col_origin=0,
                        highlights={},
                        highlight_subtensor={},
                        rank_label=True,
                        coord_label=None):

        # Default payload

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
            self.draw_label(row_origin, col_origin+col_hack, "Rank: "+self._getId(fiber))

            for c in range(fiber.getShape(all_ranks=False)[0]):
                self.draw_label(row_origin+1, col_origin+col_hack+c, f"{c:^3}")

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
                self.draw_label(row_origin, col_origin+col_hack, "...")
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

            self.draw_label(row_origin+rank_label_offset, col_origin, label)
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

        payload = 0

        #
        # Set up the highlighting for this level
        #
        highlight_coords = {}

        for worker, points in highlights.items():
            highlight_coords[worker] = [c[0] for c in points]

        for coord in range(shape[0]):
            color_coord = set([worker for worker, coords in highlight_coords.items() if coord in coords])
            color_subtensor = set([worker for worker in highlight_subtensor.keys()])
            color_coord_or_subtensor = color_coord | color_subtensor

            if isinstance(fiber, Fiber):
                payload = fiber.getPayload(coord)

            row_count = self.draw_value(row_cur, col_cur, payload, color_coord_or_subtensor)

            # row_cur does not change
            row_max = max(row_max, row_cur+row_count)

            col_cur += 1
            col_max = col_cur

            if col_max > self.col_extent: break


        return [row_max, col_max]

#
# Utility methods
#
    def _compute_next_highlights(self, c, highlights, highlight_subtensor):
        """_compute_next_highlights

        Given the current hightlights and the subtensor bulk highlight information
        compute the "next" highlight information

        Parameters:

        c: coordinate
        Current coordinate

        highlights: dictionary
        Dictionary of workers with list of points to highlight

        highlight_subtensor: list
        List of workers to highlight in all subtensors

        """

        #
        # Compute highlighting for this level
        #
        highlight_coords = {}

        for worker, points in highlights.items():
            highlight_coords[worker] = [c[0] for c in points]

        #
        # These variables hold the highlight information with one
        # less coordinate, and a list of workers that are
        # highlighting the remaining subtensor
        #
        # TBD: Code copied from tree_image...
        #
        highlight_next = {}
        highlight_subtensor_next = {}

        for worker, points in highlights.items():
            #
            # Calculate relevant points after this level
            #
            highlight_next[worker] = [ p[1:] for p in points if len(p) > 1 and p[0] == c ]

            #
            # Once we start highlighting a fiber, highlight the entire subtensor.
            # TBD: Maybe we should have just copied highlight_subtensor
            #
            if worker in highlight_subtensor:
                highlight_subtensor_next[worker] = True

            #
            # If there are no more coordinates, maybe start highlighting a subtensor 
            #
            if len(highlight_next[worker]) == 0 and c in highlight_coords[worker]:
                highlight_subtensor_next[worker] = True

        return (highlight_next, highlight_subtensor_next)


    def _getId(self, fiber):
        """ _getId - get fiber's rank id """

        if fiber.getOwner() is None:
            return ""

        return str(fiber.getOwner().getName())

#
# Image methods
#
    def image_setup(self):

        # Constrain image size (overage matches crop above)

        x_pixels = self.col2x(self.col_extent+1) + 200 # was 8192
        y_pixels = self.row2y(self.row_extent+1) + 20 # was 1024

        # Create an image at least this tall (in pixels)
        self.max_y = 100


        # Do image related setup
        self.im = Image.new("RGB", (x_pixels, y_pixels), "wheat")
        self.fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
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
    def draw_label(self, row, column, label):
        """draw_label"""

        x1 = self.col2x(column) + 20
        y1 = self.row2y(row) - 10

        # Hack: drawing text twice looks better in PIL
        self.draw.text((x1+10,y1+10), label, font=self.fnt, fill="black")
        self.draw.text((x1+10,y1+10), label, font=self.fnt, fill="black")



    def draw_value(self, row, column, value, highlight=[]):
        """draw_value"""

        #
        # Check if we're outside the box
        #
        if row >= self.row_extent or column >= self.col_extent:
            if row == self.row_extent or column == self.col_extent:
                self.draw_label(row, column, "...")
                return 2

            return 0


        if isinstance(value, Payload):
            value = value.value

        if not isinstance(value, tuple):
            value = ( value, )

        row_count = len(value)

        font_y = 30

        x1 = self.col2x(column) + 20
        y1 = self.row2y(row) - 10

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
                if v >= 10:
                    x_text = x_text - 7
                if v >= 100:
                    x_text = x_text - 7
                if v >= 1000:
                    x_text = x_text - 7


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
    def col2x(self, col):
        return 200 + 40*col
            
    def row2y(self, row):
        return 40 + 40*row


if __name__ == "__main__":
                         
    print("a - multiple highlights")
    a = Tensor("examples/data/sparse-matrix-a.yaml")
    a.setColor("blue")
    i = UncompressedImage(a, highlights=[(0,1), (1,2), (3,)])
    i.show()

    #
    print("a - single highlights")
    i = UncompressedImage(a, (1,2))
    i.show()

    #
    print("b")
    b = Tensor.fromUncompressed(["X"], [1, 2, 0, 0, 4])
    i = UncompressedImage(b, [(1,), (4,)])
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
