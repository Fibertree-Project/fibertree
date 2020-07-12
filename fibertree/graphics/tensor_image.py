from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from .tree_image import TreeImage
from .uncompressed_image import UncompressedImage

class TensorImage():
    """TensorImage

    Class to create images of a tensor or fiber. Basically a
    trampoline to the desired style, except when multiple images need
    to be combined.

    """

    def __init__(self, object, *args, highlights={}, style='tree', **kwargs):
        """__init__

        Create an image corresponding the a given tensor or fiber in
        style "style". Optionally highlight elements of the
        tensor/fiber

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

        highlights = self.canonicalizeHighlights(highlights)

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
        self.im.show()


    @staticmethod
    def canonicalizeHighlights(highlights, worker="PE"):
        """canonicalizeHighlights

        In methods that accept highlights there is considerable
        flexibility in the form that the highlights are provided. This
        method converts any of those forms into the canonical form,
        using keyword "worker" to assign a worker if one isn't
        provided in the "highlights" argument.  The canonical form is
        a dictionary of workers and lists of their highlighted points:


        {worker0: [(point0_coord0, point0_coord1, ...),
                   (point1_coord0, point1_coord1, ...),
                    ...],
         worker1: [(point0_coord0, point0_coord1, ...),
                   (point1_coord0, point1_coord1, ...),
                   ...],
         ...,
        }


        Alternative forms:

        1) Single point per worker

        {worker0: (point0_coord0, point0_coord1, ...),
         worker1: (point0_coord0, point0_coord1, ...),
          ...
        }


        2) List of points, no worker

        [(point0_coord0, point0_coord1, ...),
         (point1_coord0, point1_coord1, ...),
         ...]


        3) Single point, no worker

        (point1_coord0, point1_coord1, ...)


        Warning: if a coordinate is a tuple there is ambiguity in forms
        1 and 3, so they cannot be used.


        Parameters:
        -----------

        highlights: dictionary, list or tuple
        A specification of highlights, maybe not in canonical form

        worker: string
        A name to use for the worker, if highlights doesn't include one

        Returns:
        --------

        highlights: dictionary
        A specification of highlights in canonical form


        Raises:
        -------

        Nothing

        """

        if not isinstance(highlights, dict):
            #
            # Massage highlights into proper form
            #
            highlights = {worker: highlights}
            

        #
        # Wrap highlights specified as a single point into a list
        #
        for pe, pe_highlights in highlights.items():
            #
            # If highlights is a single point convert to list
            #
            if len(pe_highlights):
                try:
                    temp = pe_highlights[0][0]
                except Exception:
                    temp = pe_highlights
                    pe_highlights = []
                    pe_highlights.append(temp)
                    highlights[pe] = pe_highlights

        return highlights


if __name__ == "__main__":

    a = Tensor.fromYAMLfile("../../examples/data/draw-a.yaml")
    a.print()
    i = TensorImage(a)
    i.show()
