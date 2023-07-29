"""Tensor Image Module"""

import logging

from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from .highlights import HighlightManager

from .tree_image import TreeImage
from .uncompressed_image import UncompressedImage

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.tensor_image')


class TensorImage():
    """TensorImage

    Class to create images of a tensor or fiber. Basically a
    trampoline to the desired style, except when multiple images need
    to be combined.

    Constructor
    -----------

    Create an image corresponding the a given tensor or fiber in style
    "style". Optionally highlight elements of the tensor/fiber


    Parameters
    ----------
    object: tensor or fiber
        A tensor or fiber object to draw

    highlights: dictionary or list or tuple
        A dictionary of "workers" each with list of points to highlight
        list is a list of point tuples to highlight (assumes one "worker")
        tuple is a single point to highlight (assumes one "worker")

    style: string or list
        String containing "tree", "uncompressed" or
        "tree+uncompressed" indicating the style of the image to create

    extent: tuple
        Maximum row/col to use for image

    **kwargs: keyword arguments
        Additional keyword arguments to pass on to the desired style

    """

    def __init__(self, object, *args, highlights={}, style='tree', **kwargs):
        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.tensor_image')


        highlights = HighlightManager.canonicalizeHighlights(highlights)

        #
        # Conditionally unwrap Payload objects
        #
        object = Payload.get(object)

        #
        # Canonicalize the specification of the style
        #
        style = TensorImage.canonicalizeStyle(style)

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
        # TBD: Allow style to be a list
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
                im1_xoffset = -diff//2
                im2_xoffset = 0

            im.paste(im1, (im1_xoffset, 0))
            im.paste(im2, (im2_xoffset, im1.height))

            self.im = im
        else:
            print(f"TensorImage: Unsupported image style - {style}")


    def show(self):
        """Display the image

        Parameters
        ----------
        None

        """

        self.im.show()


    @staticmethod
    def canonicalizeStyle(style, count=None):
        """"Canonicalize the style

        Convert `style` into canonical display style(s) by expanding
        abbreviations, e.g., 'u' -> 'uncompressed'.

        If `count` is `None`, return a single style (as a string), and
        if `count` is an integer return a list of styles of length
        `count`.

        If `count` is an integer and `style` is a single string
        replicate it into a list of the length `count`.

        And, if `style` is a list of less than length `count`, then
        expand it to length `count` by replicating the last item in
        the given list.

        Parameters
        ----------

        style: string or list
            The user provided style or list of styles

        count: integer
            The length of the list of styles to create


        Returns
        --------

        canonical_style: string or list
            The input style(s) in canonical form

        """

        abbrevs = {
            'u': 'uncompressed',
            't': 'tree',
            't+u': 'tree+uncompressed'
        }

        if isinstance(style, str):
            style = abbrevs.get(style, style)

            if count is not None:
                style = count * [style]
        else:
            style = [abbrevs.get(s, s) for s in style]

            last_style = style[-1]

            for _ in range(len(style), count):
                style.append(last_style)

        return style


if __name__ == "__main__":

    a = Tensor.fromYAMLfile("../../examples/data/draw-a.yaml")
    a.print()
    i = TensorImage(a)
    i.show()
