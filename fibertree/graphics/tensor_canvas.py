import copy
from collections import namedtuple

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from .image_utils import ImageUtils
from .highlights import HighlightManager

from .movie_canvas import MovieCanvas
from .spacetime_canvas import SpacetimeCanvas

class TensorCanvas():
    """TensorCanvas

    The TensorCanvas class is a frontend for the
    {Movie,SpaceTime}Canvas classes. It creates a canvas of the
    requested animation type and shadows of the tracked tensors and
    passes method calls on to the created class.

    It manges the shadows of the tracked tensors in an addActivity()
    method. That method is used to support incremental addition of
    highlights on a per worker basis and handles "skew". In specific,
    addActivity() uses the shadows of the tracked tensors, logs
    highlights and logs changes to mutable tensors. Then when a frame
    ends, collects the appropriate highlights and replays the changes
    into the appropriate shadow tensor and passes the highlights to
    the addFrame() method in a {Movie,SpaceTime}Canvas, which displays
    those highlignts on the current state of the shadow tensors.

    """

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

        #
        # Places to collect information about the frames
        #
        num_tensors = len(tensors)

        self.num_tensors = num_tensors
        self.orig_tensors = []
        self.shadow_tensors = []

        for t in tensors:
            #
            # Build list of orignal tensors,
            # but convert Fibers into a Tensor with the Fiber as its root
            #
            if isinstance(t, Fiber):
                # TBD: We do not really know if the fiber is mutable...
                t = Tensor.fromFiber(fiber=t)

            self.orig_tensors.append(t)

            #
            # Create a tensor to hold a shadow tensor that tracks
            # updates to the tracked tensors at the right time
            #
            if t.isMutable():
                self.shadow_tensors.append(copy.deepcopy(t))
            else:
                self.shadow_tensors.append(t)

        self.log = []
        self.inframe = False

        #
        # Reset image highlighting
        #
        ImageUtils.resetColors()

        #
        # Create desired canvas
        #
        if animation == 'movie':
            self.canvas = MovieCanvas(*self.shadow_tensors, style=style)
        elif animation == 'spacetime':
            self.canvas = SpacetimeCanvas(*self.shadow_tensors)
        else:
            print(f"TensorCanvas: No animation type: {animation}")


    def addActivity(self, *highlights, worker="anon", skew=0, end_frame=False):
        """ addActivity """
        #
        # Set that we are in a frame
        #
        self.inframe = True

        #
        # Canonicalize highlights
        #
        # Note: The highlights parameter is a list of points or list
        # of lists of points one for each tracked tensor They will be
        # turned into an actual highlight data strcture here
        #
        highlights_list = []

        for hl in highlights:
            highlights_list.append(HighlightManager.canonicalizeHighlights(hl, worker=worker))


        #
        # Tell the canvas to remember the current tensor states
        #
        self._logChanges(*highlights_list, skew=skew)

        #
        # Collect the highlights for this frame accounting for skew
        #
        # TBD: There must be a better way to combine the highlights.
        #      Using this code exactly one addActivity() must have all
        #      the activity for a worker
        #
        for n, highlight in enumerate([highlights[worker] for highlights in highlights_list]):
            self.log[skew].highlights[n][worker] = highlight

        #
        # Sometimes addActivity should end the frame
        #
        if end_frame or worker == "anon":
            self.addFrame()


    def addFrame(self, *highlights):
        """addFrame"""

        #
        # For situations where caller did not use addActivity()
        # call it one time for them
        #
        # Note: highlights is a list of highlights objects
        #
        if highlights:
            self.addActivity(*highlights, worker="PE")

        self.inframe = False

        #
        # Highlights were collected by addActivity
        #
        highlights = self.log[0].highlights if len(self.log) else {}

        #
        # Populate shadow tensors with values for this frame
        #
        # Note: The log gets popped, so we needed to get the
        # highlights out before this call
        #
        self._replayChanges()

        #
        # Add the frame
        #
        self.canvas.addFrame(*highlights)


    def getLastFrame(self, message=None):
        """ getLastFrame """

        #
        # Push out any remaining logged activity
        #
        for n in range(len(self.log)):
            self.addFrame()

        return self.canvas.getLastFrame(message=message)


    def saveMovie(self, filename=None):
        """ saveMovie """

        #
        # Push out any remaining logged activity
        #
        for n in range(len(self.log)):
            self.addFrame()

        return self.canvas.saveMovie(filename=filename)

#
# Utility function to log and replay a series of changes to the
# tensors being tracked
#

    def _logChanges(self, *highlights, skew=0):
        """logChanges

        Log current values (at the highlighted points) to the mutable
        tensors for later replay into the shadow tensors at time
        "skew".

        Parameters:

        highlights: a highlights dictionary
        A per PE list of highlighted points for each tracked tensor

        skew: integer
        The relative time at which these values are to be replayed

        """

        tensors = self.orig_tensors

        for n in range(len(self.log), skew+1):
            self._createChanges()

        points = self.log[skew].points
        values = self.log[skew].values

        for tnum, highlight in enumerate(highlights):
            #
            # Skip immutable tensors
            #
            if not tensors[tnum].isMutable():
                continue

            #
            # Log the points being highlighted
            #
            for worker, highlight_list in highlight.items():
                for point in highlight_list:
                    if not isinstance(point, tuple):
                        point = (point,)

                    points[tnum].append(point)

                    payload = tensors[tnum].getPayload(*point)
                    values[tnum].append(copy.deepcopy(payload))



    def _replayChanges(self):
        """replayChanges """

        if len(self.log) == 0:
            return

        points = self.log[0].points
        values = self.log[0].values

        for shadow, point_list, value_list in zip(self.shadow_tensors,
                                                  points,
                                                  values):

            if shadow.isMutable():
                for point, value in zip(point_list, value_list):

                    if Payload.isEmpty(value):
                        continue

                    ref = shadow.getPayloadRef(*point)
                    ref <<= value


        del self.log[0]


    def _createChanges(self):
        """ _createChanges """

        FrameLog = namedtuple('FrameLog', ['points', 'values', 'highlights'])

        num_tensors = self.num_tensors

        new_points = [[] for n in range(num_tensors)]
        new_values = [[] for n in range(num_tensors)]
        new_highlights = [{} for n in range(num_tensors)]

        self.log.append(FrameLog(new_points, new_values, new_highlights))


#
# Utility class to manage cycles
#
class CycleManager():
    """CycleManager

    A class to allow a program to manage the current cycle, for using
    in canvas displays

    TBD: Allow nested parallel regions

    """

    def __init__(self):
        """__init__

        Initialize some variables

        """

        self.cycle = 0
        self.parallel = 0
        self.worker_max = 0


    def __call__(self):
        """__call__

        Call the class to return the current cycle and move to the
        next cycle

        """
        cycle = self.cycle
        self.cycle += 1

        return cycle


    def startParallel(self):
        """startParallel

        Start a parallel region by remembering the current cycle

        """

        self.parallel = self.cycle


    def startWorker(self):
        """startWorker

        Reset the cycle for a worker

        """

        self.cycle = self.parallel


    def finishWorker(self):
        """finishWorker

        Remember the maximum cycle (actually the cycle after any
        activity in that worker) arrived at by any worker in the
        parallel region.

        """

        self.worker_max = max(self.worker_max, self.cycle)


    def finishParallel(self):
        """finishParallel

        Finish the parallel region and set the current cycle to the
        cycle after the longest running worker

        """

        self.cycle = self.worker_max
