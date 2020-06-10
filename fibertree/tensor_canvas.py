
from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

from fibertree.movie_canvas import MovieCanvas
from fibertree.spacetime_canvas import SpacetimeCanvas

from fibertree import TensorImage

class TensorCanvas():
    """TensorCanvas"""

    def __init__(self, *tensors, animation='movie', style='tree'):
        """__init__"""

        if animation == 'movie':
            self.canvas = MovieCanvas(*tensors, style=style)
        elif animation == 'spacetime':
            self.canvas = SpacetimeCanvas(*tensors, style=style)
        else:
            print(f"TensorCanvas: No animation type: {animation}")
    

    def addFrame(self, *highlighted_coords_per_tensor):
        """ addFrame """

        self.canvas.addFrame(*highlighted_coords_per_tensor)


    def getLastFrame(self, message=None):
        """ getLastFrame """

        return self.canvas.getLastFrame(message=message)


    def saveMovie(self, filename=None):
        """ saveMovie """

        return self.canvas.saveMovie(filename=filename)
