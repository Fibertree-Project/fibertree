from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

class TensorMatrixImage():
    """TensorMatrixImage"""

    def __init__(self, object, highlights=[]):
        """__init__"""
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
        # Conditionally unwrap Payload objects
        #
        object = Payload.get(object)

        self.create_matrix(object, highlights)


    def create_matrix(self, object, highlights=[]):
        """create_matrix: Create an image of a tensor or fiber tree

        Parameters
        ----------
        object: tensor or fiber
        A tensor or fiber object to draw
        highlight: list of points (each point is a list of coordinates)
        Points in the tensor to highlight

        Returns
        -------
        Nothing

        Notes
        ------

        The drawing is made in a coordinate space where the X
        and Y are the positions in the matrix.
        Translation to pixels happens in the draw_*() methods.

        """

        #
        # Create the objects for the image
        #
        self.image_setup()

        #
        # Display either the root of a tensor or a raw fiber
        #
        if isinstance(object, Tensor):
            root = object.getRoot()
            self._color = object.getColor()
        elif isinstance(object, Fiber):
            root = object
            self._color = "red"
        else:
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
            region_size = self.traverse(root, highlights=highlights)

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
# Method to traverse (and draw) all the cells in the matrix
#
    def traverse(self, fiber, highlights=[]):
        """traverse"""

        #
        # Assume this is a matrix
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
            shape = fiber.getShape()

            if len(shape) == 3:
                region_size = self.traverse_cube(shape, fiber, highlights=highlights)
            elif len(shape) == 2:
                region_size = self.traverse_matrix(shape, fiber, highlights=highlights)
            elif len(shape) == 1:
                region_size = self.traverse_vector(shape, fiber, highlights=highlights)
            else:
                region_size = [1, 1]

        return region_size

    def traverse_cube(self, shape, fiber, row_origin=0, col_origin=0, highlights=[], highlight_subtree=False):
        """ traverse_cube - unimplemented """

        row_cur = row_origin
        row_max = row_origin
        col_cur = col_origin

        #
        # Set up the highlighting for this level
        #
        highlight_coords = [ c[0] for c in highlights ]


        for matrix_c in range(shape[0]):
            matrix_p = fiber.getPayload(matrix_c)

            highlight_next = [ p[1:] for p in highlights if len(p) > 1 and p[0] == matrix_c ]

            # Once we start highlighting a fiber, highlight the entire subtensor.
            highlight_payload = highlight_subtree
            if len(highlight_next) == 0:
                highlight_payload |= matrix_c in highlight_coords

            rc_range = self.traverse_matrix(shape[1:],
                                            matrix_p,
                                            row_origin=row_cur,
                                            col_origin=col_cur,
                                            highlights=highlight_next,
                                            highlight_subtree=highlight_payload)

            row_max = max(row_max, rc_range[0])
            col_cur = rc_range[1] + 2
            
        return [ row_max, col_cur ]



    def traverse_matrix(self, shape, fiber, row_origin=0, col_origin=0, highlights=[], highlight_subtree=False):
        """ traverse_matrix """

        #
        # Print out the rank information (if available)
        #
#        if col == 0 and not fiber.getOwner() is None:
#            self.draw_rank(level, "Rank: %s " % fiber.getOwner().getName())


        #
        # Set up the highlighting for this level
        #
        highlight_coords = [ c[0] for c in highlights ]

        #
        # Set up variables to track rows and columns
        row_c = row_origin
        row_max = row_origin
        row_p = Fiber([], [])

        col_max = col_origin

        for count in range(shape[0]):

            if fiber is not None:
                row_p = fiber.getPayload(row_c)

            highlight_next = [ p[1:] for p in highlights if len(p) > 1 and p[0] == row_c ]

            # Once we start highlighting a fiber, highlight the entire subtensor.
            highlight_payload = highlight_subtree
            if len(highlight_next) == 0:
                highlight_payload |= row_c in highlight_coords

            rc_range = self.traverse_vector(shape[1:],
                                             row_p,
                                             row_origin=row_c,
                                             col_origin=col_origin,
                                             highlights=highlight_next,
                                             highlight_subtree=highlight_payload)

            row_max = max(row_max, rc_range[0])
            row_c = row_max

            col_max = max(col_max, rc_range[1])

        return [ row_max, col_max ]


    def traverse_vector(self, shape, fiber, row_origin=0, col_origin=0, highlights=[], highlight_subtree=False):
        # Default payload

        row_cur = row_origin
        row_max = row_origin

        col_p = 0
        highlight_coords = [ c[0] for c in highlights ]

        for col_c in range(shape[0]):
            if fiber is not None:
                col_p = fiber.getPayload(col_c, default=0)

            lightitup = (col_c in highlight_coords) or highlight_subtree
            row_count = self.draw_value(row_origin, col_origin+col_c, col_p, lightitup)

            row_max = max(row_max, row_cur+row_count)

        return [ row_max, col_origin+shape[0] ]

#
# Image methods
#
    def image_setup(self):

        # TBD: Estimate image size based on size of tensor

        # Create an image at least this tall (in pixels)
        self.max_y = 100

        # Do image related setup
        self.im = Image.new("RGB", (8192, 1024), "wheat")
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
    def draw_value(self, row, column, value, highlight=False):
        """draw_value"""

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
        return 40+ 40*row


if __name__ == "__xmain__":
                         
    a = Tensor("examples/data/sparse-matrix-a.yaml")
    a.setColor("blue")
    a.print()
    i = TensorMatrixImage(a, highlights=[(0,1), (1,2), (3,)])
    i.show()

    #
    i = TensorMatrixImage(a, (1,2))
    i.show()

    #
    b = Tensor.fromUncompressed(["X"], [1, 2, 0, 0, 4])
    b.print()
    i = TensorMatrixImage(b, [(1,), (4,)])
    i.show()

if __name__ == "__main__":
    a = Tensor("examples/data/sparse-matrix-a.yaml")

    #
    a_root = a.getRoot()
    c = Tensor.fromFiber(["X", "Y", "Z"], Fiber([0, 1, 2], [a_root, Fiber([],[]), a_root]))
    c.print()
    i = TensorMatrixImage(c)
    i.show()

    #
    d = c.getRoot()
    print("Original")
    i = TensorMatrixImage(d)
    i.show()

    #
    d_flattened = d.flattenRanks()
    print("Flattened")
    i = TensorMatrixImage(d_flattened)
    i.show()
