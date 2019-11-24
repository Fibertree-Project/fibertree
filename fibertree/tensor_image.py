from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber

class TensorImage():
    """TensorImage"""

    def __init__(self, tensor):
        """__init__"""

        self.create_tree(tensor)


    def create_tree(self, tensor):

        self.image_setup()

        f = tensor.root()

        region_start = 0
        region_end = self.traverse(f)
        region_size = region_end - region_start
        
        #
        # Draw root of tree
        #
        self.draw_rank(0, "File: %s" % tensor.yamlfile)

        fiber_size = 1
        fiber_start = region_start + (region_size - fiber_size)/2
        self.draw_coord(0, fiber_start, "R")
        self.draw_line(0, region_size/2, 1, region_size/2)

        self.im = self.im.crop((0,
                                0,
                                self.offset2x(region_end)+200,
                                self.level2y(2+len(tensor.rank_ids))))

        
    def show(self):
        self.im.show()

        
    def traverse(self, fiber, level=1, offset=0):
        """traverse"""

        if offset == 0:
#           print("(%02d, 00) - Rank - %s" % (level, fiber.owner.name))
            self.draw_rank(level, "Rank: %s " % fiber.owner.name)

        region_start = offset
        region_end = region_start

        #
        # Figure out space of region below this fiber
        #

        targets = []
        coordinate_start = region_start
        
        for (c, p) in fiber:
            if isinstance(p, Fiber):
                region_end = self.traverse(p, level+1, region_end)
            else:
                region_end += 1

            targets.append(coordinate_start+(region_end-coordinate_start)/2)
            coordinate_start = region_end

        region_size = region_end - region_start
#       print("(%02d, %02d) - %0d - Region" % (level, region_start, region_size))

        #
        # Display fiber for this level
        #
        fiber_size = len(fiber)
        fiber_start = region_start + (region_size - fiber_size)/2


#       print("(%02d, %02d) - %0d - Fiber" % (level, fiber_start, fiber_size))
        self.draw_fiber(level, fiber_start, fiber_start+fiber_size)

        pos=fiber_start

        for c,p in fiber:
#           print("(%02d, %02d) - Coord - %02d" % (level, pos, c))
            self.draw_coord(level, pos, c)
            self.draw_line(level, pos+0.5, level+1, targets.pop(0))

            if not isinstance(p, Fiber):
#               print("(%02d, %02d) - Value - %02s" % (level+1, pos, p))
                self.draw_value(level+1, pos, p.value)

            pos += 1


        return region_end

#
# Image methods
#
    def image_setup(self):

        self.im = Image.new("RGB", (2048, 512), "wheat")
        self.fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        self.draw = ImageDraw.Draw(self.im)
        

    def draw_rank(self, level, rank):
        """draw_rank"""

        x1 = 0
        y1 = self.level2y(level)

        self.draw.text((x1+10,y1+10), rank, font=self.fnt, fill="black")

    def draw_fiber(self, level, start, end):
        """draw_fiber"""

        height = 60
        gap = 5

        x1 = self.offset2x(start) + gap
        y1 = self.level2y(level) - 10
        x2 = self.offset2x(end) - gap
        y2 = y1 + height
        
        self.draw.ellipse(((x1,y1), (x2,y2)), (128,128,128), 1)


    def draw_coord(self, level, offset, coord):
        """draw_coord"""

        x1 = self.offset2x(offset) + 20
        y1 = self.level2y(level)
        x2 = x1 + 40
        y2 = y1 + 40
        
        self.draw.ellipse(((x1,y1), (x2,y2)), "blue", 1)
        self.draw.text((x1+15,y1+10), str(coord), font=self.fnt, fill="white")

    def draw_value(self, level, offset, value):
        """draw_value"""

        x1 = self.offset2x(offset) + 20
        y1 = self.level2y(level) - 10
        x2 = x1 + 40
        y2 = y1 + 40
        
        self.draw.rectangle(((x1,y1), (x2,y2)), "red", 1)
        self.draw.text((x1+10,y1+10), str(value), font=self.fnt, fill="black")

    def draw_line(self, level1, offset1, level2, offset2):

        # Bottom of source is 40 below level2y result (see draw_coord)
        # Top of target is 10 above level2y results (see draw_fiber)

        x1 = self.offset2x(offset1)
        y1 = self.level2y(level1) + 40
        x2 = self.offset2x(offset2)
        y2 = self.level2y(level2) - 10

        self.draw.line([ (x1, y1), (x2, y2) ], width=3, fill="black")


    def offset2x(self, offset):
        return 200 + 80*offset
            
    def level2y(self, level):
        return 40+ 80*level


if __name__ == "__main__":

    a = Tensor("examples/data/draw-a.yaml")
    a.print()
    i = TensorImage(a)
    i.show()
