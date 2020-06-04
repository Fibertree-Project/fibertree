import copy

from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

from fibertree import TensorImage

class SpacetimeCanvas():
    """SpaceTimeCanvas"""

    def __init__(self, *tensors, style='tree'):
        """__init__"""

        #
        # Set image type
        #
        self.style = style

        #
        # Conditionally unwrap Payload objects
        #
        self.tensors = []
        self.spacetime = []
        self.highlights = []

        for tensor in tensors:
            self.tensors.append(Payload.get(tensor))

            spacetime = Tensor(rank_ids=["T", "S"])
            if isinstance(tensor, Tensor):
                spacetime.setColor(tensor.getColor())

            self.spacetime.append(spacetime)
            self.highlights.append([])

        self.frame = 0


    def addFrame(self, *highlighted_coords_per_tensor):
        """addFrame"""

        #
        # Handle the case where nothing should be highlighted anywhere.
        #
        final_coords = []
        if not highlighted_coords_per_tensor:
            for n in range(len(self.tensors)):
              final_coords.append([])
        else:
            final_coords = highlighted_coords_per_tensor

        # TBD: Should fiber append get the root if you try to append a tensor

        for tensor, spacetime, highlights, hl_coords in zip(self.tensors, self.spacetime, self.highlights, final_coords):

            timestep = tensor.getRoot()
            timestep_ranks = len(timestep.getShape(all_ranks=True))

            if timestep_ranks > 1:
                timestep = timestep.flattenRanks(levels=timestep_ranks-1)

            spacetime.getRoot().append(self.frame, copy.deepcopy(timestep))

            highlight_t = list(hl_coords)
            highlight_t.append(self.frame)

            highlights.append(highlight_t)

        self.frame += 1


    def getLastFrame(self):
        """getLastFrame"""

        images = []

        for spacetime, highlights in zip(self.spacetime, self.highlights):

            if isinstance(spacetime, Tensor):
                # Get the root out of tensor
                spacetime_root = spacetime.getRoot()
            else:
                # A fiber is itself the root
                spacetime_root = spacetime

            shape = spacetime_root.getShape(all_ranks=True)

            if len(shape) == 1:
                #
                # Original tensor was a vector
                #
                highlights_mapped = highlights
            else:
                #
                # Original tensor was a matrix or bigger
                #
                #
                # Build a map of spatial tensor points to flattened positions
                #
                # Note: points in the tensor look like (time, coord0, coord1, ..)
                #       so we need to skip over the first rank
                #
                point2pos = {}

                for position, (point, value)  in enumerate(spacetime_root[-1].payload):
                    point2pos[point] = position

                print(f"Point to position mapping:  {point2pos}")

                #
                # Map the highlights into the new flattened space
                #
                # Note: highlights look like: (coord0, coord1, ..., time)
                #       and need to look like: (position, time)
                #
                highlights_mapped = []

                for h in highlights:
                    h1 = tuple(h[0:-1])
                    h2 = h[-1]
                    try:
                        h12 = (point2pos[h1], h2)
                        highlights_mapped.append(h12)
                    except:
                        print(f"Could not map point ({h1},{h2}) in point2pos array")


                #
                # Flatten the names of the coordinates in the spacetime tensor
                #
                spacetime_root.updateCoords(lambda i, c, p: i, depth=1)

            #
            # Swap the space and time ranks
            #
            spacetime_swapped = spacetime.swapRanks()

            #
            # Create spacetime image for this tensor and append to full image
            #
            image = TensorImage(spacetime_swapped,
                                style='uncompressed',
                                highlights=highlights_mapped).im

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
