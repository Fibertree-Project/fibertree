"""Movie Canvas Module"""

import logging
import numpy
import cv2
import copy

from PIL import Image, ImageDraw, ImageFont
from tqdm.notebook import tqdm

from fibertree import Tensor
from fibertree import TensorImage
from fibertree import UncompressedImage
from fibertree import Fiber
from fibertree import Payload
from fibertree import ImageUtils

from .canvas_layout import CanvasLayout

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.movie_canvas')


class MovieCanvas():
    """MovieCanvas

    A class to create a movie of activity in a set of tensors. This
    class is used by the `TensorCanvas` class as one of the ways it
    can display activity. Various ways of displaying the tenor (e.g.,
    `TreeImage` and `UncompressedImage`) are supported.


    Constructor
    -----------

    Parameters
    ----------
    tensors: list
        A list of tensors or fibers objects to track

    style: string (default: 'tree')
        Display style ('tree', 'uncompressed', 'tree+uncompressed')

    progress: Boolean (default: True)
        Enable tqdm style progress bar on movie creation

    title: string(defalt: "")
        A title for the movie

    layout: list (default: [len(tensors)*[1]]
        List of the number of tensors in each row
    
    """

    def __init__(self,
                 *tensors,
                 style='tree',
                 title="",
                 layout=None,
                 progress=True):

        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.movie_canvas')

        #
        # Set various image attributes
        #
        self.title = title
        self.style = TensorImage.canonicalizeStyle(style, count=len(tensors))
        self.layout = layout

        #
        # Set tqdm control
        #
        self.use_tqdm = progress
        
        #
        # Set up tensor class variables
        #
        # Note: We conditionally unwrap Payload objects
        #
        self.tensors = []
        self.image_list_per_tensor = []
        for tensor in tensors:
            self.tensors.append(Payload.get(tensor))
            self.image_list_per_tensor.append([])

        #
        # Set up per frame caption list
        #
        self.caption_list = []

        #
        # Font to use for text
        #
        self.font = ImageUtils.getFont('DejaVuSansMono', 16)

        ascent, descent = self.font.getmetrics()
        self.font_height = ascent + descent

       #
        # Add an initial frame with nothing highlighted (it looks good)
        #
        self.addFrame()


    def addFrame(self, *highlighted_coords_per_tensor, caption=""):
        """Add a frame to the movie

        Create an image of each tracked tensor preperly highlighted and
        append those images to the "per frame" lists associated with the
        tracked tensor.

        Also remember the "caption" to be associated with this frame by
        including it in the "per frame" list of captions.

        Parameters
        ----------

        highlighted_coords_per_tensor: list of highlights
            Highlights to add to the registered tensors

        caption: string
            The caption associated with this frame


        Note: This method must be called in frame order. Dealing with
              any out-of-order of creation of frames must be handled
              before this method is called, e.g., in TensorCanvas:addActivity

        """

        #
        # Handle the case where nothing should be highlighted anywhere.
        #
        if not highlighted_coords_per_tensor:
            final_coords = [[] for n in range(len(self.tensors))]
        else:
            final_coords = highlighted_coords_per_tensor

        assert len(final_coords) == len(self.tensors)

        for n in range(len(self.tensors)):
            tensor = self.tensors[n]
            highlighted_coords = final_coords[n]
            style = self.style[n]

            im = TensorImage(tensor,
                             style=style,
                             highlights=highlighted_coords).im

            self.image_list_per_tensor[n].append(im)

        self.caption_list.append(caption)

    def getLastFrame(self, message=None):
        """Get the final frame

        Get an image of the final frame. This method also adds a final
        frame with nothing highlighted, because it looks better

        Parameters
        ---------

        message: string, default=None
            A message to add to the image

        Returns
        -------
        final_frame: image
            An image of the final frame

        """

        #
        # Force creation of the final frame
        #
        end = len(self.image_list_per_tensor[0])
        (final_images, final_width, final_height) = self._combineFrames(end-1, end)

        if message is None:
            return final_images[-1]

        #
        # Add message to final image
        #
        im = final_images[-1].copy()

        ImageDraw.Draw(im).text((15, final_height-65),
                                message,
                                font=self.font,
                                fill="black")

        return im


    def getAllFrames(self, layout=None):

        (final_images, _, _) = self._combineFrames(layout=layout)

        return final_images


    def saveMovie(self, filename=None, layout=None):
        """Save the movie to a file

        Parameters
        ----------

        filename: string, default=None
            Name of a file to save the movie

        """

        (final_images, final_width, final_height) = self._combineFrames(layout=layout)

        fourcc = cv2.VideoWriter_fourcc(*"vp09")
        out = cv2.VideoWriter(filename, fourcc, 1, (final_width, final_height))

        tqdm_desc = "Render video frame for each cycle"

        for image in self._tqdm(final_images, desc=tqdm_desc):
            for duplication_cnt in range(1):
                out.write(cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR))

        out.release()

#
# Internal utility functions
#
    def _combineFrames(self, start=0, end=None, layout=None):

        if end is None:
            end = len(self.image_list_per_tensor[0])

        #
        # Obtain the shape of each tensors for the frames 
        #
        if layout is None:
            layout = self.layout

        canvas_layout = CanvasLayout(self.image_list_per_tensor, self.layout)

        (core_width, core_height, tensor_shapes) = canvas_layout.getLayout(start, end)

        #
        # Dump individual frames into the same image so they stay in sync.
        #
        final_width = core_width

        header_height = 75

        max_captions = max([len(captions) for captions in self.caption_list])
        footer_height = 150 + max_captions * self.font_height

        final_height = header_height + core_height + footer_height

        final_images = []

        tqdm_desc = "Paste individual tensor images into frame for each cycle"

        for n in self._tqdm(range(start, end), desc=tqdm_desc):
            
            #
            # Create empty frame for pasting tensor images into
            #
            final_images.append(Image.new("RGB",
                                          (final_width, final_height),
                                          "wheat"))

            #
            # Populat4e the image for this timestep
            #
            current_tensor = 0
            row_x = 0
            row_y = 80         # Leave room for title

            for row_width, row_height, tensor_widths in tensor_shapes:

                #
                # Center row of tensor images in full image
                #
                row_x = final_width // 2 - row_width // 2 

                for tensor_width in tensor_widths:

                    image = self.image_list_per_tensor[current_tensor][n]

                    #
                    # Center individual tensor in its cell
                    #
                    tensor_x_left = row_x + tensor_width // 2  - image.width // 2

                    final_images[n-start].paste(image, (tensor_x_left, row_y))

                    row_x += tensor_width
                    current_tensor += 1

                row_y += row_height
            
        #
        # Add cycle information to the images
        # (skipping extra frames at beginning and end)
        #
        for n, im in enumerate(final_images[1:-1]):
            #
            # Draw title
            #
            title = self.title

            ImageDraw.Draw(im).text((15, 5),
                                    title,
                                    font=self.font,
                                    fill="black")

            #
            # Draw footer (cycle info and captions)
            #
            footer = self._createFooter(n, self.caption_list[n+1])

            ImageDraw.Draw(im).text((15, final_height - footer_height),
                                    footer,
                                    font=self.font,
                                    fill="black")

        return (final_images, final_width, final_height)

#
# Utitlity functions
#
    def _createFooter(self, cycle, captions):
        #
        # Create common, per_pe tuples
        # And find length of common spacerspacer
        #
        caption_tuples = []
        max_common = 0

        for caption in captions:
            if " & " in caption:
                common, per_pe = caption.split(" & ", 1)
            else:
                common = caption
                per_pe = ""

            caption_tuples.append((common, per_pe))

            max_common = max(max_common, len(common))

        #
        # Create list of footers
        #
        cycle_prefix = f"Cycle: {cycle} - "
        cycle_spacer = len(cycle_prefix) * ' '
        common_spacer = max_common * ' '

        result = []

        #
        # First line includes the cycle info and common part
        #
        common, per_pe = caption_tuples[0]
        result.append(f"{cycle_prefix}{common:<{max_common}} {per_pe}")
        last_common = common

        #
        # Following lines optionally start a new common section
        #
        for common, per_pe in caption_tuples[1:]:

            if common != last_common:
                result.append(f"{cycle_spacer}{common:<{max_common}} {per_pe}")
                last_common = common
            else:
                result.append(f"{cycle_spacer}{common_spacer} {per_pe}")

        return "\n".join(result)

#
# Tqdm-related methods
#
# TBD: Move to some more central location
#

    def _tqdm(self, iterable, desc=""):
        """
        _tqdm

        Conditional tqdm based on whether we are in a notebook

        """

        if self.use_tqdm and MovieCanvas._in_ipynb():
            return tqdm(iterable, desc=desc, leave=False)
        else:
            return iterable


    @staticmethod
    def _in_ipynb():
        """
        _in_ipynb

        Are we in an IPython notebook?

        """

        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True
            else:
                return False
        except NameError:
            return False



if __name__ == "__main__":

    a = Tensor.fromYAMLfile("../../examples/data/draw-a.yaml")
    b = Tensor.fromYAMLfile("../../examples/data/draw-b.yaml")
    canvas = MovieCanvas(a, b)
    canvas.addFrame()
    canvas.addFrame([10], [4])
    canvas.addFrame([10, 40], [4, 1])
    canvas.addFrame([10, 40, 1], [4, 1, 0])
    canvas.addFrame()
    canvas.saveMovie("/tmp/tmp.mp4")
    print("Try playing /tmp/tmp.mp4")
