from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

class UncompressedImage():
    """UncompressedImage"""


    def __init__(self, object, highlights=[], extent=(30, 200)):
        """__init__

        Parameters
        ----------

        object: tensor or fiber
        A tensor or fiber object to draw

        highlight: list of points (each point is a list of coordinates)
        Points in the tensor to highlight

        extent: tuple
        Maximum row/col to use for image

    """

        #
        # If highlights is a single point convert to list
        #
        if len(highlights):
            try:
                temp = highlights[0][0]
            except Exception:
                temp = highlights
                highlights = []
                highlights.append(temp)

        #
        # Record paramters
        #
        # Note: We conditionally unwrap Payload objects
        #
        self.object = Payload.get(object)
        self.highlights = highlights
        self.row_extent = extent[0]
        self.col_extent = extent[1]

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


    def traverse_cube(self, shape, fiber, row_origin=1, col_origin=0, highlights=[], highlight_subtree=False):
        """ traverse_cube - unimplemented """

        #
        # Print out the rank information (if available)
        #
        self.draw_label(row_origin, col_origin, "Rank: "+self._getId(fiber)+" ----->")

        row_cur = row_origin + 1
        row_max = row_origin + 1
        col_cur = col_origin

        #
        # Set up the highlighting for this level
        #
        highlight_coords = [ c[0] for c in highlights ]


        #
        # Just show the nonEmpty matrices
        #
        for matrix_c, matrix_p in fiber:

            highlight_next = [ p[1:] for p in highlights if len(p) > 1 and p[0] == matrix_c ]

            # Once we start highlighting a fiber, highlight the entire subtensor.
            highlight_payload = highlight_subtree
            if len(highlight_next) == 0:
                highlight_payload |= matrix_c in highlight_coords

            self.draw_label(row_origin, col_cur+5, f"{matrix_c}")

            rc_range = self.traverse_matrix(shape[1:],
                                            matrix_p,
                                            row_origin=row_cur,
                                            col_origin=col_cur,
                                            highlights=highlight_next,
                                            highlight_subtree=highlight_payload)

            # row_cur does not change
            row_max = max(row_max, rc_range[0])

            col_cur = rc_range[1] + 2
            col_max = col_cur

            if col_cur > self.col_extent: break


        return [row_max, col_max]



    def traverse_matrix(self, shape, fiber, row_origin=1, col_origin=0, highlights=[], highlight_subtree=False):
        """ traverse_matrix """

        #
        # Print out the rank information (if available)
        #
        label = "Rank: "+self._getId(fiber)
        self.draw_label(row_origin+1, col_origin, label)

        #
        # Set up variables to track rows and columns (note offset for rank label)
        #
        row_cur = row_origin
        row_max = row_cur

        col_cur = col_origin + (len(label)+2)//3
        col_max = col_cur

        #
        # Set up the highlighting for this level
        #
        highlight_coords = [ c[0] for c in highlights ]


        #
        # Set up for loop
        #
        row_p = Fiber([], [])
        row_first = True

        for row_c in range(shape[0]):

            if fiber is not None:
                row_p = fiber.getPayload(row_c)

            highlight_next = [ p[1:] for p in highlights if len(p) > 1 and p[0] == row_c ]

            # Once we start highlighting a fiber, highlight the entire subtensor.
            highlight_payload = highlight_subtree
            if len(highlight_next) == 0:
                highlight_payload |= row_c in highlight_coords

            rc_range = self.traverse_vector(shape[1:],
                                             row_p,
                                             row_origin=row_cur,
                                             col_origin=col_cur,
                                             highlights=highlight_next,
                                             highlight_subtree=highlight_payload,
                                             label=row_first)

            row_max = max(row_max, rc_range[0])
            row_cur = row_max
            row_first = False

            # col_cur does not change
            col_max = max(col_max, rc_range[1])

            if row_cur > self.row_extent: break


        return [row_max, col_max]


    def traverse_vector(self, shape, fiber, row_origin=1, col_origin=0, highlights=[], highlight_subtree=False, label=True):
        # Default payload

        #
        # Print out the rank information (if available)
        #
        if label:
            self.draw_label(row_origin, col_origin, "Rank: "+self._getId(fiber))
            label_offset = 1
        else:
            label_offset = 0

        #
        # Set up variables to track rows and columns (note offset for rank label)
        #
        row_cur = row_origin + label_offset
        row_max = row_origin + label_offset

        col_cur = col_origin
        col_max = col_origin

        payload = 0

        highlight_coords = [c[0] for c in highlights]

        for coord in range(shape[0]):
            col_cur = col_origin + coord

            if isinstance(fiber, Fiber):
                payload = fiber.getPayload(coord)

            lightitup = (coord in highlight_coords) or highlight_subtree
            row_count = self.draw_value(row_cur, col_cur, payload, lightitup)

            # row_cur does not change
            row_max = max(row_max, row_cur+row_count)
            # col_cur is set above
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



    def draw_value(self, row, column, value, highlight=False):
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

        fill_color = self._color if value != (0, ) else 0
        fill_color = "goldenrod" if highlight else fill_color

        self.draw.rectangle(((x1,y1), (x2,y2)), fill_color, 1)

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
