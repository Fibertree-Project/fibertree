
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

#        assert(len(final_coords) == len(self.tensors))

#        for n in range(len(self.tensors)):
#            tensor = self.tensors[n]
#            highlighted_coords = final_coords[n]
#            im = TensorImage(tensor, style=self.style, highlights=highlighted_coords).im
#            self.image_list_per_tensor[n].append(im)

        # TBD: Should fiber append get the root if you try to append a tensor

        for tensor, spacetime, highlights, hl_coords in zip(self.tensors, self.spacetime, self.highlights, final_coords):

            spacetime.getRoot().append(self.frame, tensor.getRoot())

            highlight_t = list(hl_coords)
            highlight_t.append(self.frame)
            #
#            for h in hl_coords:
#                highlight_t = [ self.frame ]
#                highlight_t.append(h)

            highlights.append(highlight_t)

        self.frame += 1


    def getLastFrame(self):
        """getLastFrame"""

        images = []

        for spacetime, highlights in zip(self.spacetime, self.highlights):

            spacetime_swapped = spacetime.swapRanks()

            image = TensorImage(spacetime_swapped, style='uncompressed', highlights=highlights).im
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
