import copy

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

from fibertree import TensorImage


class SpacetimeCanvas():
    """SpaceTimeCanvas"""

    def __init__(self, *tensors):
        """__init__

        Parameters
        ----------
        tensors: list
        A list of tensors or fibers objects to track

        """

        #
        # Structures to hold infomation about each tracked tensor
        #
        self.tensors = []
        self.saved_tensors = None
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
                assert len(tensor.getShape()) == 1, "Only 1-D fibers are supported"

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


    def createSnapshot(self):
        """createSnapshot

        Hold a copy of the current state of the tracked tensors for display
        at a later time.

        """

        self.saved_tensors = []

        for tensor in self.tensors:
            #
            # TBD: Make copy conditional on whether the tensor is mutable
            #
            self.saved_tensors.append(copy.deepcopy(tensor))


    def deleteSnapshot(self):
        """deleteSnapshot"""

        self.saved_tensors = None


    def addFrame(self, *highlighted_coords_per_tensor):
        """addFrame"""

        #
        # Create snapshot if necessary
        #
        if self.saved_tensors is None:
            self.createSnapshot()

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
        for tensor, spacetime, highlights, hl_info in zip(self.saved_tensors, self.spacetime, self.highlights, final_coords):

            #
            # Get fiber holding current state
            #
            # TBD: Should fiber append get the root if you try to append a tensor
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
            # Delicate sequence to add highlight into spacetime tensor's highlight object
            #
            for worker, hl_list in hl_info.items():
                hl_list_new = []
                for point in hl_list:
                    point_list = list(point)
                    point_list.append(self.frame_num)
                    hl_list_new.append(tuple(point_list))

                if worker not in highlights:
                    highlights[worker] = hl_list_new
                else:
                    highlights[worker] = highlights[worker] + hl_list_new

        self.deleteSnapshot()
        self.frame_num += 1


    def getLastFrame(self, message=None):
        """getLastFrame"""

        images = []

        for spacetime, highlights in zip(self.spacetime, self.highlights):
            #
            # Get spacetime tensor name & ranks
            #
            #
            spacetime_name = spacetime.getName()
            spacetime_ranks = len(spacetime.getShape())

            if spacetime_ranks == 2:
                #
                # Original tensor was a vector
                #
                highlights_mapped = highlights
                pos2point = None
            else:
                #
                # Original tensor was a matrix or bigger, so flatten it
                #
                # Note: points in the tensor look like (time, coord0,
                #       coord1, ..)  so we need to skip over the first
                #       rank before flattening
                #
                spacetime = spacetime.flattenRanks(depth=1, levels=spacetime_ranks-2)
                spacetime_root = spacetime.getRoot()
                #
                #
                # Build a map of original tensor points to a scalar
                # number space, i.e., 0, 1, 2...
                #
                # Note: We rely on the fact that the last time step
                #       has all the possible points, so the scalar
                #       number space is actually the position in the
                #       final timestep.
                #
                point2pos = {}
                pos2point = {}

                for position, (point, value)  in enumerate(spacetime_root[-1].payload):
                    if isinstance(point, tuple):
                        point2pos[point] = position
                        pos2point[position] = point
                    else:
                        point2pos[(point,)] = position
                        pos2point[postion] = point

                #
                # Let user know the point mapping
                #
                #print(f"Point to position mapping:  {point2pos}")

                #
                # Remap the highlights into the new flattened space
                #
                # Note: highlights look like: (coord0, coord1, ..., time)
                #       and need to look like: (position, time)
                #
                highlights_mapped = {worker: [] for worker in highlights.keys()}

                for worker, points in highlights.items():
                    for point in points:
                        h1 = tuple(point[0:-1])
                        h2 = point[-1]
                        try:
                            h12 = (point2pos[h1], h2)
                            highlights_mapped[worker].append(h12)
                        except:
                            print(f"Could not map point ({h1},{h2}) in point2pos array")

                #
                # Remap the names of the coordinates in the spacetime tensor from
                # (coord0, coord1, ....) to a scalar.
                #
                spacetime_root.updateCoords(lambda i, c, p: point2pos[c], depth=1)

            #
            # Swap the space and time ranks
            #
            spacetime_swapped = spacetime.swapRanks()
            spacetime_swapped.setName(spacetime_name)

            #
            # Create spacetime image for this tensor and append to full image
            #
            image = TensorImage(spacetime_swapped,
                                style='uncompressed',
                                highlights=highlights_mapped,
                                row_map=pos2point).im

            images.append(image)


        return images


    def saveMovie(self):
        """saveMovie"""

        print("SpaceTimeCanvas: saveMovie - unimplemented")
        return None


if __name__ == "__main__":

    #
    # This is broken...
    #
    a = Tensor("../examples/data/draw-a.yaml")
    b = Tensor("../examples/data/draw-b.yaml")
    canvas = TensorCanvas(a, b)
    canvas.addFrame()
    canvas.addFrame([10], [4])
    canvas.addFrame([10,40], [4,1])
    canvas.addFrame([10,40,1], [4,1,0])
    canvas.addFrame()
    canvas.saveMovie("tmp.mp4")
