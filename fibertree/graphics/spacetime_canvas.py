"""Spacetime Canvas Module"""

import logging
import copy

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from fibertree import TensorImage

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.spacetime_canvas')



class SpacetimeCanvas():
    """SpaceTimeCanvas

    A class to create a spacetime diagram of activity in a set of
    tensors. This class is used by the `TensorCanvas` class as one of
    the ways it can display activity.

    Constructor
    -----------

    Parameters
    ----------
    tensors: list
        A list of tensors or fibers objects to track

    """

    def __init__(self, *tensors):
        """__init__"""

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.spacetime_canvas')

        #
        # Structures to hold infomation about each tracked tensor
        #
        self.tensors = []
        self.spacetime = []
        self.highlights = []

        for tensor in tensors:
            #
            # Append each tensor being tracked, conditionally
            # unwraping it if it is a Payload object
            #
            self.tensors.append(Payload.get(tensor))

            #
            # Create a "spacetime" tensor to hold the spacetime
            # information for this tracked tensor
            #
            if isinstance(tensor, Tensor):
                assert tensor.getShape() != [], "No support for 0-D tensors"

                spacetime = Tensor(rank_ids=["T"] + tensor.getRankIds())
                spacetime.setName(tensor.getName())
                spacetime.setColor(tensor.getColor())
            else:
                assert tensor.getDepth() == 1, "Only 1-D fibers are supported"

                spacetime = Tensor(rank_ids=["T", "S"])

            #
            # Append the "spacetime" tensor to hold this tracked
            # tensor's spacetime information
            #
            self.spacetime.append(spacetime)
            #
            # Append an empty highlight object to hold the highlighting
            # information for this tracked tensor
            #
            self.highlights.append({})

        self.frame_num = 0


    def addFrame(self, *highlighted_coords_per_tensor):
        """Add a timestep to the spacetime diagram

        Parameters
        ----------

        highlighted_coords_per_tensor: list of highlights
            Highlights to add to the registered tensors

        """

        #
        # Handle the case where nothing should be highlighted anywhere.
        #
        if not highlighted_coords_per_tensor:
            final_coords = [{} for n in range(len(self.tensors))]
        else:
            final_coords = highlighted_coords_per_tensor

        #
        # For each tracked tensor collect the information for the new frame
        #
        for tensor, spacetime, highlights, hl_info in zip(self.tensors,
                                                          self.spacetime,
                                                          self.highlights,
                                                          final_coords):

            #
            # Get fiber holding current state
            #
            # TBD: Should fiber append get the root,
            #      if you try to append a tensor
            #
            if isinstance(tensor, Tensor):
                timestep = tensor.getRoot()
            else:
                timestep = tensor

            #
            # Append current tracked tensor state to spacetime tensor
            # with a coordinate coresponding the the frame number
            #
            spacetime.getRoot().append(self.frame_num, copy.deepcopy(timestep))

            #
            # Delicate sequence to add highlight into
            # spacetime tensor's highlight object
            #
            for worker, hl_list in hl_info.items():
                hl_list_new = []
                for point in hl_list:
                    if len(point) == 1:
                        point = point[0]

                    hl_list_new.append((point, self.frame_num))

                if worker not in highlights:
                    highlights[worker] = hl_list_new
                else:
                    highlights[worker] = highlights[worker] + hl_list_new

        self.frame_num += 1


    def getLastFrame(self, message=None):
        """Get the final frame

        Create a image of the final spacetime diagram.

        Parameters
        ---------

        message: string, default=None
            A message to add to the image

        Returns
        -------
        final_frame: image
            An image of the spacetime diagram

        """

        images = []

        for spacetime, highlights in zip(self.spacetime, self.highlights):
            #
            # Get spacetime tensor name & ranks
            #
            #
            spacetime_name = spacetime.getName()
            spacetime_ranks = spacetime.getDepth()

            if spacetime_ranks > 2:
                #
                # Original tensor was a matrix or bigger, so flatten it
                #
                # Note: points in the tensor look like (time, coord0,
                #       coord1, ..)  so we need to skip over the first
                #       rank before flattening
                #
                spacetime = spacetime.flattenRanks(depth=1,
                                                   levels=spacetime_ranks-2)

            #
            # Swap the space and time ranks
            #
            spacetime_swapped = spacetime.swapRanks()
            spacetime_swapped.setName(spacetime_name)

            #
            # Create spacetime image for this tensor and append to
            # full image
            #
            image = TensorImage(spacetime_swapped,
                                style='uncompressed',
                                highlights=highlights).im

            images.append(image)

        return images


    def saveMovie(self):
        """saveMovie

        Does nothing for spacetime diagrams.

        """

        print("SpaceTimeCanvas: saveMovie - unimplemented")
        return None


if __name__ == "__main__":

    #
    # This is broken...
    #
    a = Tensor.fromYAMLfile("../examples/data/draw-a.yaml")
    b = Tensor.fromYAMLfile("../examples/data/draw-b.yaml")
    canvas = TensorCanvas(a, b)
    canvas.addFrame()
    canvas.addFrame([10], [4])
    canvas.addFrame([10, 40], [4, 1])
    canvas.addFrame([10, 40, 1], [4, 1, 0])
    canvas.addFrame()
    canvas.saveMovie("tmp.mp4")
