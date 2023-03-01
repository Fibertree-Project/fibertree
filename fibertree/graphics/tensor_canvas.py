"""Tensor Canvas Module"""

import logging

import copy
from collections import namedtuple

from fibertree import Tensor
from fibertree import Fiber
from fibertree import Payload

from .image_utils import ImageUtils
from .highlights import HighlightManager

from .movie_canvas import MovieCanvas
from .spacetime_canvas import SpacetimeCanvas

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.tensor_canvas')


class TensorCanvas():
    """TensorCanvas

    The TensorCanvas class is a frontend for the
    {Movie,SpaceTime}Canvas classes. It creates a canvas of the
    requested animation type and shadows of the tracked tensors and
    passes method calls on to the created class.

    It manges the shadows of the tracked tensors in an addActivity()
    method. That method is used to support incremental addition of
    highlights at a specfic time and at a particular worker, which is
    is specified by the "spacetime" keyword. In more detail,
    addActivity() uses the shadows of the tracked tensors, logs
    highlights and logs changes to mutable tensors. Then when a frame
    is to be displayed it collects the appropriate highlights and
    replays the changes into the appropriate shadow tensor and passes
    highlights to the addFrame() method in a {Movie,SpaceTime}Canvas,
    which displays those highlights on the current state of the shadow
    tensors.

    The addFrame() method can be called exclictly to output the
    information from the oldest frame, but it is best to just wait
    until the information for all frames has been recorded, and all
    frames will be output..

    This class also provides primitive support for having an activity
    "wait" for a coordinate in another tensor to be updated. To do
    this it tracks the update time of each coordinate that changes
    a value. This capability is enabled with the "enable_wait" keyword.

    Constructor
    -----------

    Create an animation canvas of the requested type for for the given
    tensors.

    Parameters
    ----------
    tensors: list
        A list of tensors or fiber objects to track

    animation: string
        Type of animation ('none', 'movie', 'spacetime')

    style: string
        Display style for movies ('tree', 'uncompressed', 'tree+uncompressed')

    enable_wait: Boolean
        Enable tracking update times to allow waiting for an update

    """

    def __init__(self, *tensors, animation='movie', style='tree', enable_wait=False):
        """__init__

        """
        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.tensor_canvas')

        #
        # Places to collect information about the frames
        #
        num_tensors = len(tensors)

        self.num_tensors = num_tensors
        self.orig_tensors = []
        self.shadow_tensors = []

        self.using_spacetimestamp = None

        self.update_times = [] if enable_wait else None
        self.waitname_map = {}

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

            #
            # Create a tensor to hold last update time
            # of each element of a tensor
            #
            if enable_wait:
                self.update_times.append(Tensor(rank_ids=t.getRankIds()))

        #
        # Create a list to hold a record of activity at each timestamp
        #
        self.log = []

        #
        # Flag to help addFrame() know if it is adding activity
        # to an existing frame
        #
        self.inframe = False

        #
        # Global cycle tracker
        #
        self.cycle = 0

        #
        # Reset image highlighting
        #
        ImageUtils.resetColors()

        #
        # Create desired canvas
        #
        # Note: We create the canvas with the shadow tensors, so that
        # the visualized activity happens in the desired order
        #
        if animation == 'movie':
            self.canvas = MovieCanvas(*self.shadow_tensors, style=style)
        elif animation == 'spacetime':
            self.canvas = SpacetimeCanvas(*self.shadow_tensors)
        elif animation == 'none':
            self.canvas = NoneCanvas(*self.shadow_tensors, style=style)
        else:
            self.logger.warning("TensorCanvas: No animation type: %s", animation)


    def addActivity(self, *highlights, spacetime=None, worker="anon", skew=0, wait=None, end_frame=False):
        """addActivity

        Add an activity to be displayed in an animation.

        Parameters
        ----------
        highlights: list
            A list of highlight specifications for each tensor being animated
            See highlights.py for formats for highlights.

        spacetime: tuple
            A tuple containing the "worker" performing the activity and a "timestamp"
            specifying the time the activity occurs. Timestamps are tuples of integers

        worker: string
            Name of the worker performing the action (mutually exclusive with `spacetime`)

        skew: integer
            Time the activity occurs relative to current time (mutually exclusive with `spacetime`)

        wait: list of tensors
            Specify a list of tensors that must be updated before this activiy can occur. After
            the dependency is satisfied add the skew.

        end_frame: Boolean
            If true call addFrame() after adding activity. Deprecated.


        Notes
        -----

        For mutable tensors the `highlights` parameter must
        authoritatively indicate the **points** that have been changed
        for the first time.

        """
        #
        # Don't need to do anything for `NoneCanvas`
        #
        if isinstance(self.canvas, NoneCanvas):
            return

        #
        # Rename spacetime to spacetimestamp to avoid confusion
        # with the spacetime style
        #
        spacetimestamp = spacetime

        #
        # Set that we are in a frame
        #
        self.inframe = True

        if spacetimestamp is not None:
            #
            # Spacetimestamp mode
            #
            assert self.using_spacetimestamp in [None, True], "One cannot mix spacetimestamp and skew"

            self.using_spacetimestamp = True

            worker = spacetimestamp[0]
            timestamp = spacetimestamp[1]
        else:
            #
            # Skew mode - user must invoke AddFrame()
            #
            assert self.using_spacetimestamp in [None, False], "One cannot mix spacetimestamp and skew"

            self.using_spacetimestamp = False

            #
            # Convert skew into a global time
            #
            timestamp = self.cycle + skew

        #
        # Canonicalize highlights
        #
        # Note 1: The highlights parameter is a list of points or list
        # of lists of points one for each tracked tensor They will be
        # turned into an actual highlight data strcture here.
        #
        # Note 2: The highlight specification must not contain a worker,
        # because that will override the default worker, but is not checked.
        #
        highlights_list = []

        for hl in highlights:
            highlights_list.append(HighlightManager.canonicalizeHighlights([hl], worker=worker))

        #
        # If wait is a list it is a list of input tensors that this
        # activity depended on and the skew is delayed by the latest
        # time among those inputs
        #
        if wait is not None:
            assert False, "Wait is currently broken"
            assert self.update_times is not None, "Keyword 'enable_wait' not set"

            delay = -1

            #
            # Look at each input and see which is the latest
            #
            # TBD: We wait for all the highlighted points in an input,
            #      maybe it should be selective
            #
            for tname, xmit_time in wait.items():

                tnum = self.waitname_map.get(tname, self._insertWaitname(tname))

                for hl in highlights_list[tnum][worker]:
                    update_time = self.update_times[tnum].getPayload(*hl)
                    update_delay = update_time.value - self.cycle + xmit_time
                    if update_delay > delay:
                        delay = update_delay

            assert delay >= 0, "Tensor never updated for wait"

            skew = max(skew, delay)

        #
        # Tell the canvas to remember the current tensor states
        #
        log_idx = self._logChanges(*highlights_list, timestamp=timestamp)

        #
        # Collect the highlights for this frame accounting for global time
        #
        # TBD: There must be a better way to combine the highlights.
        #      Using this code exactly one addActivity() must have all
        #      the activity for a worker
        #
        active_highlights = self.log[log_idx].highlights

        for n, highlight in enumerate([highlights[worker] for highlights in highlights_list]):

            if worker not in active_highlights[n]:
                active_highlights[n][worker] = highlight
                self.logger.debug("New highlight %s", highlight)
                self.logger.debug("After: %s", active_highlights[n][worker])
            else:
                self.logger.debug("Before: %s", active_highlights[n][worker])
                self.logger.debug("Appending highlight %s", highlight)
                active_highlights[n][worker].extend(highlight)
                self.logger.debug("After: %s", active_highlights[n][worker])

        #
        # Sometimes addActivity should end the frame
        #
        if end_frame or worker == "anon":
            self.addFrame()


    def _insertWaitname(self, tname):

        if isinstance(tname, int):
            self.waitname_map[tname] = tname
            return tname

        for tnum, t in enumerate(self.orig_tensors):
            if t.getName() == tname:
                self.waitname_map[tname] = tnum
                return tnum

        #
        # Didn't find the tensor, so
        # return None to cause an error in the caller
        #
        return None


    def addFrame(self, *highlights):
        """Add a step to the movie or spacetime diagram

        A step (or cycle) to the animation. For movies this
        corresponds to a frame in the movie.

        Parameters
        ----------

        highlighted_coords_per_tensor: list of highlights
            Highlights to add to the registered tensors

        """
        #
        # Don't need to do anything for `NoneCanvas`
        #
        if isinstance(self.canvas, NoneCanvas):
            return


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
        """Finalize the movie/spacetime diagram

        Finalize the animation by adding all the pending cycles to the
        animation.  Get an image of the final frame.

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
        # Push out any remaining logged activity
        #
        for n in range(len(self.log)):
            self.addFrame()

        return self.canvas.getLastFrame(message=message)


    def saveMovie(self, filename=None):
        """Save the animation to a file

        If the animation can be saved to a file, this method will do
        that.

        Parameters
        ----------

        filename: string, default=None
            Name of a file to save the movie

        """

        #
        # Push out any remaining logged activity
        #
        self.getLastFrame()

        return self.canvas.saveMovie(filename=filename)

#
# Utility function to log and replay a series of changes to the
# tensors being tracked
#

    def _logChanges(self, *highlights, timestamp=None):
        """logChanges

        Log current values (at the highlighted points) to the mutable
        tensors for later replay into the shadow tensors at time
        "timestamp".

        Parameters
        ----------

        highlights: a highlights dictionary
            A per PE list of highlighted points for each tracked tensor

        timestamp: tuple of integers
            The time at which these values are to be replayed

        """

        assert timestamp is not None, "Timestamp error"

        tensors = self.orig_tensors
        update_times = self.update_times

        #
        # Find the log entry for "timestamp" or create one
        #
        log_idx_list = [ idx for idx, element in enumerate(self.log) if element.timestamp == timestamp]
        if len(log_idx_list) >= 1:
            log_idx = log_idx_list[0]
            self.logger.debug("Found existing timestamp at %s", log_idx)
        else:
            log_idx = self._createChanges(timestamp)

        #
        # Get references to the lists of points and values updated at timestamp
        #
        points = self.log[log_idx].points
        values = self.log[log_idx].values

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

                    if update_times is not None:
                        updatetime_ref = update_times[tnum].getPayloadRef(*point)
                        updatetime_ref <<= timestamp

        return log_idx


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

        #
        # Increment cycle
        #
        if not self.using_spacetimestamp:
            self.cycle += 1


    def _createChanges(self, timestamp):
        """ _createChanges """

        FrameLog = namedtuple('FrameLog', ['timestamp', 'points', 'values', 'highlights'])

        num_tensors = self.num_tensors

        new_points = [[] for n in range(num_tensors)]
        new_values = [[] for n in range(num_tensors)]
        new_highlights = [{} for n in range(num_tensors)]

        framelog = FrameLog(timestamp, new_points, new_values, new_highlights)

        if len(self.log) == 0:
            self.log.append(framelog)
            return 0

        #
        # Insert new changes at proper place in the log
        #   TBD: Do a more sophisticated insertion
        #

        for i in range(len(self.log)):
            if self.log[i].timestamp > timestamp:
                log_idx = i
                self.log.insert(log_idx, framelog)
                return log_idx

        self.log.append(framelog)

        return len(self.log)-1


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


class NoneCanvas():
    """NoneCanvas - does nothing"""

    def __init__(self, *tensors, animation='movie', style='tree', **kwargs):
        """__init__"""

        # For 'none' we create a movie but don't add any frames
        self.canvas = TensorCanvas(*tensors, animation='movie', style=style, **kwargs)

        return

    def addFrame(self, *highlighted_coords_per_tensor):
        """addFrame - should never get called"""

        return

    def getLastFrame(self, message=None):
        """getLastFrame"""

        im = self.canvas.getLastFrame(message=message)

        return im


    def saveMovie(self, filename=None):
        """saveMovie"""

        self.logger.info("NoneCanvas: saveMovie - unimplemented")
        return None
