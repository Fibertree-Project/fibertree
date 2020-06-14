
from fibertree import Tensor
from fibertree import Fiber
from fibertree.payload import Payload

from fibertree.movie_canvas import MovieCanvas
from fibertree.spacetime_canvas import SpacetimeCanvas

from fibertree import TensorImage

class TensorCanvas():
    """TensorCanvas"""

    def __init__(self, *tensors, animation='movie', style='tree'):
        """__init__

        Parameters
        ----------
        tensors: list
        A list of tensors or fibers objects to track

        animation: string
        Type of animation ('none', 'movie', 'spacetime')

        style: string
        Display style for movies ('tree', 'uncompressed', 'tree+uncompressed')

        """

        if animation == 'movie':
            self.canvas = MovieCanvas(*tensors, style=style)
        elif animation == 'spacetime':
            self.canvas = SpacetimeCanvas(*tensors)
        else:
            print(f"TensorCanvas: No animation type: {animation}")

        #
        # Place to collect highlights from each worker
        #
        self.inframe = False
        self.num_tensors = len(tensors)
        self.frame_highlights = [{} for n in range(self.num_tensors)]


    def addActivity(self, *highlights, worker="anon", end_frame=False):
        """ addFrame """
        #
        # Note: The highlights parameter is a list of points or list
        # of lists of points one for each tracked tensor They will be
        # turned into an actual highlight data strcture by addFrame
        #
        self.inframe = True

        #
        # Tell the canvas to remember the current tensor states
        #
        self.canvas.createSnapshot()

        #
        # Collect the highlights for this frame
        #
        for n, highlight in enumerate(highlights):
            self.frame_highlights[n][worker] = highlight

        if end_frame or worker == "anon":
            self.addFrame()



    def addFrame(self, *highlights):
        """addFrame"""

        # Note: highlights is a list of highlights objects

        if self.inframe:
            #
            # Highlights were collected by addActivity
            #
            highlights = self.frame_highlights

            self.inframe = False
            self.frame_highlights = [{} for n in range(self.num_tensors)]
        else:
            #
            # Unpack parameter into a list
            #
            highlights = list(highlights)

        #
        # Canonicalize highlights
        #
        new_highlights = []

        for hl in highlights:
            new_highlights.append(TensorImage.canonicalizeHighlights(hl))

        self.canvas.addFrame(*new_highlights)


    def getLastFrame(self, message=None):
        """ getLastFrame """

        #
        # Push out the last addFrame if it is still hanging around
        #
        if self.inframe:
            self.addFrame()

        return self.canvas.getLastFrame(message=message)


    def saveMovie(self, filename=None):
        """ saveMovie """

        #
        # Push out the last addFrame if it is still hanging around
        #
        if self.inframe:
            self.addFrame()

        return self.canvas.saveMovie(filename=filename)
