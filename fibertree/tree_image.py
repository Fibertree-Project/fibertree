from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

class TreeImage():
    """TreeImage"""


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

        self.create_tree()


    def create_tree(self):
        """create_tree: Create an image of a tensor or fiber tree

        Notes
        ------

        The drawing is made in a coordinate space where the X
        dimension is measured in the number of non-empty fiber
        coordinates being displayed and the Y dimension is measured in
        layers of the tree. Translation to pixels happens in the
        draw_*() methods.

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
            # Note: a rank_id may be a list that needs to be convert to a string
            #
            ranks = ", ".join([ str(r) for r in object.getRankIds()])

            if name:
                self.draw_rank(0, f"Tensor: {name}[{ranks}]")
            else:
                self.draw_rank(0, f"File: {object.yamlfile}")
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
            self.draw_coord(0, 0, "R")
            self.draw_line(0, 1/2, 1, 1/2)
            self.draw_value(1, 0, Payload.get(root))
            region_end = 1
        else:
            #
            # Draw a non-0-D tensor or a fiber, i.e., the fiber tree
            #
            region_end = self.traverse(root, highlights=highlights)

        #
        # Crop the image
        #
        self.im = self.im.crop((0,
                                0,
                                self.offset2x(region_end)+200,
                                20+self.max_y))


    def show(self):
        self.im.show()


#
# Method to traverse (and draw) all the levels of the tree
#
    def traverse(self, fiber, level=0, offset=0, highlights=[], highlight_subtensor=False):
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
                self.draw_coord(0, 0, "R")
                self.draw_line(0, 1/2, 1, 1/2)
                self.draw_value(1, 0, Payload.get(fiber))
                region_end = 1
            else:
                #
                # Recursively traverse and draw the fibers of a non-0-D tensor
                #
                region_end = self.traverse(fiber, level=1, offset=offset, highlights=highlights, highlight_subtensor=highlight_subtensor)
                region_size = region_end - region_start
                #
                # Draw root of tree
                #
                fiber_size = 1
                fiber_start = region_start + (region_size - fiber_size)/2
                self.draw_coord(0, fiber_start, "R")
                self.draw_line(0, region_size/2, 1, region_size/2)

            return region_end

        #
        # Process the fibers of the tree (level > 0)
        #

        #
        # Print out the rank information (if available)
        #
        if offset == 0 and not fiber.getOwner() is None:
            self.draw_rank(level, "Rank: %s " % fiber.getOwner().getName())

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
        # Set up the highlighting for this level
        #
        highlight_coords = [ c[0] for c in highlights ]

        for n, (c, p) in enumerate(fiber):
#            if n > 10: break

            if Payload.contains(p, Fiber):
                highlight_next = [ p[1:] for p in highlights if len(p) > 1 and p[0] == c ]

                # Once we start highlighting a fiber, highlight the entire subtensor.
                highlight_payload = highlight_subtensor
                if len(highlight_next) == 0:
                    highlight_payload |= c in highlight_coords

                region_end = self.traverse(Payload.get(p), level+1, region_end, highlight_next, highlight_payload)
            else:
                region_end += 1

            targets.append(coordinate_start+(region_end-coordinate_start)/2)
            coordinate_start = region_end


        region_size = region_end - region_start



        #
        # Display fiber for this level
        #
        fiber_size = len(fiber)
        fiber_start = region_start + (region_size - fiber_size)/2

        self.draw_fiber(level, fiber_start, fiber_start+fiber_size, highlight_subtensor)

        pos=fiber_start

        for c,p in fiber:
            self.draw_coord(level, pos, c, (c in highlight_coords) or highlight_subtensor)
            if (c in highlight_coords) and not highlight_subtensor:
                self.draw_intra_line(level, fiber_start + fiber_size / 2, pos+0.5, True)

            # Only draw the line if the next level will actually draw something.
            if not Payload.contains(p, Fiber) or not p.isEmpty():
                self.draw_line(level, pos+0.5, level+1, targets.pop(0), (c in highlight_coords) or highlight_subtensor)

            if not Payload.contains(p, Fiber):
                highlight_payload = c in highlight_coords    # How could this not be the leaf --- "and rest_of_highlighting == []"
                self.draw_value(level+1, pos, Payload.get(p), highlight_payload or highlight_subtensor)

            pos += 1

        return region_end

#
# Image methods
#
    def image_setup(self):

        # Constrain image size (overage matches crop above)

        x_pixels = self.offset2x(self.col_extent+1) + 200 # was 8192
        y_pixels = self.level2y(self.row_extent+1) + 20 # was 1024

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
#          - level: layer in the tree (Y)
#          - offset: number of drawn fiber coordinates (X)
#
    def draw_rank(self, level, rank):
        """draw_rank"""

        x1 = 0
        y1 = self.level2y(level)

        # Hack: drawing text twice looks better in PIL
        self.draw.text((x1+10,y1+10), rank, font=self.fnt, fill="black")
        self.draw.text((x1+10,y1+10), rank, font=self.fnt, fill="black")


    def draw_fiber(self, level, start_offset, end_offset, highlight=False):
        """draw_fiber"""

        height = 60
        gap = 5

        x1 = self.offset2x(start_offset) + gap
        y1 = self.level2y(level) - 10
        x2 = self.offset2x(end_offset) - gap
        y2 = y1 + height
        fill_color = (128,128,128) if not highlight else (233,198,109)
        
        self.draw.ellipse(((x1,y1), (x2,y2)), fill_color, (0, 0, 0))


    def draw_coord(self, level, offset, coord, highlight=False):
        """draw_coord"""

        x1 = self.offset2x(offset) + 20
        y1 = self.level2y(level)
        x2 = x1 + 40
        y2 = y1 + 40
        color = "goldenrod" if highlight else "black"
        x_text = x1+15
        if coord != "R" and isinstance(coord, int):
            if int(coord) >= 10:
                x_text = x_text - 7
            if int(coord) >= 100:
                x_text = x_text - 7
            if int(coord) >= 1000:
                x_text = x_text - 7

        self.draw.ellipse(((x1,y1), (x2,y2)), color, 1)
        # Hack: drawing text twice looks better in PIL
        self.draw.text((x_text,y1+10), str(coord), font=self.fnt, fill="white")
        self.draw.text((x_text,y1+10), str(coord), font=self.fnt, fill="white")


    def draw_value(self, level, offset, value, highlight=False):
        """draw_value"""

        if isinstance(value, Payload):
            value = value.value

        if not isinstance(value, tuple):
            value = ( value, )

        font_y = 30

        x1 = self.offset2x(offset) + 20
        y1 = self.level2y(level) - 10

        x2 = x1 + 40
        y2 = y1 + len(value)*(font_y+10)
        
        if y2 > self.max_y:
            self.max_y = y2

        fill_color = "goldenrod" if highlight else self._color

        self.draw.rectangle(((x1,y1), (x2,y2)), fill_color, 1)

        for i,v in enumerate(value):
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



    def draw_line(self, level1, offset1, level2, offset2, highlight=False):

        # Bottom of source is 40 below level2y result (see draw_coord)
        # Top of target is 10 above level2y results (see draw_fiber)

        x1 = self.offset2x(offset1)
        y1 = self.level2y(level1) + 40
        x2 = self.offset2x(offset2)
        y2 = self.level2y(level2) - 10
        fill_color = "goldenrod" if highlight else "black"

        self.draw.line([ (x1, y1), (x2, y2) ], width=3, fill=fill_color)

    def draw_intra_line(self, level, fiber_offset, coord_offset, highlight=False):

        # Bottom of source is 10 above level2y results (see draw_line)
        # Top of target is level2y result (see draw_coord)

        x1 = self.offset2x(fiber_offset)
        y1 = self.level2y(level) - 10
        x2 = self.offset2x(coord_offset)
        y2 = self.level2y(level)
        fill_color = "goldenrod" if highlight else "black"

        self.draw.line([ (x1, y1), (x2, y2) ], width=3, fill=fill_color)

#
#
# Methods to convert positions specified in offset/level space into pixels
#
    def offset2x(self, offset):
        return 200 + 80*offset
            
    def level2y(self, level):
        return 40+ 80*level


if __name__ == "__main__":

    a = Tensor("examples/data/draw-a.yaml")
    a.print()
    i = TreeImage(a)
    i.show()
