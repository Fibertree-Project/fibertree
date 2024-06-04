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

    layout: list (default: [len(tensors)*[1]]
        List of the number of tensors in each row
    
    """

    def __init__(self, *tensors, style='tree', progress=True, layout=None):
        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.movie_canvas')

        #
        # Set image type
        #
        self.style = TensorImage.canonicalizeStyle(style, count=len(tensors))

        #
        # Set tqdm control
        #
        self.use_tqdm = progress

        #
        # Set up pattern
        #

        layout_tensors = sum(layout)
        total_tensors = len(tensors)
        
        # Add rows to cover all tensors
        if layout_tensors  < total_tensors:
            layout.extend([1] * (total_tensors - layout_tensors))
        
        self.layout = layout
        
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
        # Font to use for text
        #
        self.font = ImageUtils.getFont('DejaVuSansMono', 16)

        #
        # Add an initial frame with nothing highlighted (it looks good)
        #
        self.addFrame()


    def addFrame(self, *highlighted_coords_per_tensor):
        """Add a frame to the movie

        Parameters
        ----------

        highlighted_coords_per_tensor: list of highlights
            Highlights to add to the registered tensors

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
        # Create a frame with nothing highlighted
        #
        self.addFrame()

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


    def saveMovie(self, filename=None):
        """Save the movie to a file

        Parameters
        ----------

        filename: string, default=None
            Name of a file to save the movie

        """

        end = len(self.image_list_per_tensor[0])
        (final_images, final_width, final_height) = self._combineFrames(0, end)

        fourcc = cv2.VideoWriter_fourcc(*"vp09")
        out = cv2.VideoWriter(filename, fourcc, 1, (final_width, final_height))

        for image in self._tqdm(final_images):
            for duplication_cnt in range(1):
                out.write(cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR))

        out.release()

#
# Internal utility functions
#
    def _combineFrames(self, start, end):

        #
        # Obtain the shape of each tensors for the frames 
        #
        (final_width, final_height, tensor_shapes) = self._calcShapes(start, end)

        #
        # Dump individual frames into the same image so they stay in sync.
        #
        final_images = []

        for n in range(start, end):
            
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
            row_y = 0

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
        for n, im in enumerate(final_images[1:]):
            message = f"Cycle: {n}"

            ImageDraw.Draw(im).text((15, final_height-80),
                                    message,
                                    font=self.font,
                                    fill="black")

        return (final_images, final_width, final_height)

    def _calcShapes(self, start, end):

        layout = self.layout

        tensor_dims = self._calcTensorSizes()

        final_width = 0
        final_height = 0
        row_shapes = []

        current_tensor = 0
        
        for row_length in layout:

            if current_tensor >= len(tensor_dims):
                break

            #
            # Extract out tensor dimensions for tensors in this row
            #
            row_dims = tensor_dims[current_tensor:current_tensor+row_length]

            #
            # Calculate the width/height of row
            # Record width of each tensor in the row
            # Track total width and height
            #
            row_width = 0
            row_height = 0
            row_widths = []
            
            for tensor_width, tensor_height in row_dims:
            
                row_width += tensor_width
                row_height = max(row_height, tensor_height)

                row_widths.append(tensor_width)


            final_width = max(final_width, row_width)
            final_height += row_height
            row_shapes.append([row_width, row_height, row_widths])

            current_tensor += row_length

        #
        # Add a little padding at the bottom for when the controls are visible.
        #
        final_height = final_height + 75

        return (final_width, final_height, row_shapes)
    
    
    def _calcTensorSizes(self):
        """_finalize"""

        #
        # For each tensor find the image of that tensor with the maximum width and height
        #
        tensor_dims = []

        for t in range(len(self.tensors)):
            max_width = 0
            max_height = 0
            for image in self.image_list_per_tensor[t]:
                max_height = image.height if (image.height > max_height) else max_height
                max_width  = image.width  if (image.width  > max_width)  else max_width

            tensor_dims.append((max_width, max_height))

        return tensor_dims

#
# Tqdm-related methods
#
# TBD: Move to some more central location
#

    def _tqdm(self, iterable):
        """
        _tqdm

        Conditional tqdm based on wheter we are in a notebook

        """

        if self.use_tqdm and MovieCanvas._in_ipynb():
            return tqdm(iterable)
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
