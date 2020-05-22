from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

from fibertree.tree_image import TreeImage
from fibertree.uncompressed_image import UncompressedImage

class TensorImage():
    """TensorImage"""

    def __init__(self, object, *args, highlights=[], style='tree', **kwargs):
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

        #
        # Create the subimages
        #
        if "tree" in style:
            im1 = TreeImage(object, *args, highlights=highlights, **kwargs).im

        if "uncompressed" in style:
            im2 = UncompressedImage(object, *args, highlights=highlights, **kwargs).im

        #
        # Create the final image 
        #
        if style == "tree":
            self.im = im1
        elif style == "uncompressed":
            self.im = im2
        elif style == "tree+uncompressed":
            color="wheat"
            im = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)

            diff = im1.width - im2.width

            if diff > 0:
                # im1 is bigger
                im1_xoffset = 0
                im2_xoffset = diff//2
            else:
                # im2 is bigger
                im1_xoffset = diff//2
                im2_xoffset = 0

            im.paste(im1, (im1_xoffset, 0))
            im.paste(im2, (im2_xoffset, im1.height))

            self.im = im
        else:
            print(f"TensorImage: Unsupported image style - {style}")



    def show(self):
        self.im.show()



if __name__ == "__main__":

    a = Tensor("examples/data/draw-a.yaml")
    a.print()
    i = TensorImage(a)
    i.show()
