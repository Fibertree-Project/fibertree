#cython: language_level=3
class Metrics:
    """A globally available class for tracking metrics.

    The Metrics class provides an interface for collecting metrics from the
    execution of an HFA kernel. All methods are class-level so that the metrics
    can be updated and read from anywhere.

    Attributes
    ----------

    The Metrics class has a set of attributes that can be set and accessed.
    These include:

    - A **collecting** boolean, which specifies whether or not data collection
      is in progress.

    - A **metrics** list, which contains dictionaries associated with the metrics
      collected by the program.


    Constructor
    ----------

    There is no reason to ever create an instance of the Metrics class

    """
    # Create a class instance variable for the metrics collection
    collecting = False
    fiber_label = {}
    iteration = None
    line_order = None
    loop_order = None
    metrics = None
    num_cached_uses = 1000
    point = None
    prefix = None
    traces = {}
    trace_info = {}

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def addUse(cls, rank, coord, type_="iter", info=[]):
        """Add a use of all tensors at the given rank and coord

        Parameters
        ----------

        rank: str
            The name of the rank

        coord: int
            The coordinate at this rank

        type_: str
            Description of the information this trace is collecting

        info: list
            Any additional info that should be added to the CSV

        Returns
        -------

        None

        """
        assert cls.collecting

        # Update the point
        i = cls.line_order[rank]
        cls.point[i] = coord

        # Make sure we are tracking this rank and type_
        if rank not in cls.traces.keys():
            return

        if type_ not in cls.traces[rank].keys():
            return

        # Add the trace
        iteration = ",".join(str(j) for j in cls.iteration[:(i + 1)])
        point = ",".join(str(j) for j in cls.point[:(i + 1)])

        data = [iteration, point]
        if info:
            str_info = [val if isinstance(val, str) else str(val) for val in info]
            data.append(",".join(str_info))

        cls.traces[rank][type_].append(",".join(data) + "\n")

        # If we are at the limit of the number of cached uses, write the data
        # to disk
        if len(cls.traces[rank][type_]) == cls.num_cached_uses:
            cls._writeTrace(rank, type_)

    @classmethod
    def beginCollect(cls, prefix=None):
        """Begin metrics collection

        Start collecting metrics during future HFA program execution.

        Parameters
        ----------

        prefix: str
            The prefix to the files that will store the reuse statistics

        Returns
        -------

        None

        """
        cls.collecting = True
        cls.fiber_label = {}
        cls.iteration = []
        cls.line_order = {}
        cls.loop_order = []
        cls.metrics = {}
        cls.point = []
        cls.prefix = prefix
        cls.traces = {}
        cls.trace_info = {}

    @classmethod
    def dump(cls):
        """Get the most-recently collected set of metrics

        Return the dictionary containing all metrics collected since the most
        recent `Metrics.beginCollect()`

        Parameters
        ----------

        None

        Returns
        -------

        metrics: a dictionary
            The dictionary of metrics collected

        """
        return cls.metrics

    @classmethod
    def endCollect(cls):
        """End metrics collection

        Stop collecting metrics during future HFA program execution.

        Parameters
        ----------

        None

        Returns
        -------

        None

        """
        # Save the trace of uses
        for rank, dicts in cls.traces.items():
            for type_ in dicts:
                cls._writeTrace(rank, type_)

        # Clear all info
        cls.collecting = False
        cls.fiber_label = {}
        cls.iteration = None
        cls.line_order = None
        cls.loop_order = None
        cls.num_cached_uses = 1000
        cls.point = None
        cls.prefix = None
        cls.traces = {}
        cls.trace_info = {}

    @classmethod
    def endIter(cls, rank):
        """
        End iteration over a given rank

        Parameters
        ----------

        rank: str
            The name of the rank whose iteration is over

        Returns
        -------

        None

        """
        assert cls.collecting

        cls.fiber_label[rank] = 0
        cls.iteration[cls.line_order[rank]] = 0

    @classmethod
    def getIter(cls):
        """Get the inner loop iteration number

        Parameters
        ----------

        None

        Returns
        -------

        None

        """
        return tuple(cls.iteration)

    @classmethod
    def getLabel(cls, rank):
        """Get a new label for a fiber at this rank

        Parameters
        ----------

        rank: str
            The rank whose fiber we want to label

        Returns
        -------

        None

        """
        assert cls.collecting
        assert rank in cls.line_order.keys()

        cls.fiber_label[rank] += 1
        return cls.fiber_label[rank] - 1

    @classmethod
    def incCount(cls, line, metric, inc):
        """Increment a count metric during collection

        Increment the given count metric associated with the given line of
        code by the specified amount. If the line or metric is not already
        being tracked, they are first added to the dictionary of metrics.

        Parameters
        ----------

        line: string
            The name of the line number this metric is associated with

        metric: string
            The name of the metric

        inc: int
            The amount to increment the metric by.

        Returns
        -------

        None

        """
        assert cls.collecting

        line = line.strip()

        if line not in cls.metrics:
            cls.metrics[line] = {}

        if metric not in cls.metrics[line]:
            cls.metrics[line][metric] = 0

        cls.metrics[line][metric] += inc


    @classmethod
    def incIter(cls, line):
        """Increment the given line's iteration number by one

        Parameters
        ----------

        line: string
            The name of the line number this metric is associated with

        Returns
        -------

        None

        """
        assert cls.collecting

        if line not in cls.line_order.keys():
            return

        cls.iteration[cls.line_order[line]] += 1

    @classmethod
    def isCollecting(cls):
        """Returns True during metrics collection

        Returns True if metrics are being collected, and false if they are not.

        Parameters
        ----------

        None

        Returns
        -------

        None

        """
        return cls.collecting

    @classmethod
    def registerRank(cls, rank):
        """Register a rank as a part of the loop order

        Parameters
        ----------

        rank: str
            The name of the rank to register

        Returns
        -------

        None

        """
        assert cls.collecting

        # If this rank has already been registered, do nothing
        if rank in cls.line_order.keys():
            return

        cls.fiber_label[rank] = 0
        cls.iteration.append(0)
        cls.line_order[rank] = len(cls.iteration) - 1
        cls.loop_order.append(rank)
        cls.point.append(0)

        if rank in cls.traces.keys():
            for type_ in cls.traces[rank]:
                cls._startTrace(rank, type_)

    @classmethod
    def setNumCachedUses(cls, num_cached_uses):
        """Set the number of uses that are saved to memory before the trace is
        written to disk per rank

        Parameters
        ----------

        num_cached_uses: int
            Number of uses to cache

        Returns
        -------

        None

        """
        assert num_cached_uses > 1

        cls.num_cached_uses = num_cached_uses

    @classmethod
    def _startTrace(cls, rank, type_="iter"):
        """Start to trace the given rank

        Parameters
        ----------

        rank: str
            The name of the rank to register

        type_: str
            Description of the information this trace is collecting

        Returns
        -------

        None

        """
        assert rank in cls.line_order.keys()

        end = cls.line_order[rank] + 1

        with open(cls.prefix + "-" + rank + "-" + type_ + ".csv", "w") as f:
            f.write("")

        pos = ",".join(r + "_pos" for r in cls.loop_order[:end])
        coord = ",".join(r for r in cls.loop_order[:end])

        headings = [pos, coord]
        if cls.trace_info[rank][type_]:
            headings.append(",".join(cls.trace_info[rank][type_]))
        cls.traces[rank][type_].append(",".join(headings) + "\n")


    @classmethod
    def trace(cls, rank, type_="iter", info=None):
        """Set a rank to trace

        Note must be called after Metrics.beginCollect()

        Parameters
        ----------

        rank: str
            The rank to collect the trace of

        type_: str
            Description of the information this trace is collecting

        info: Optional[List[str]]
            Any additional info that should be added to the CSV

        Returns
        -------

        None

        """
        assert cls.prefix is not None
        assert cls.collecting

        if rank not in cls.traces.keys():
            cls.traces[rank] = {}
            cls.trace_info[rank] = {}

        cls.traces[rank][type_] = []

        # Default info for common types
        fields = type_.split("_")
        if fields[0] == "intersect":
            info = [fields[1] + "_match", fields[2] + "_match"]
        elif fields[0] == "populate":
            info = [fields[1] + "_access", fields[2] + "_access"]

        # Save the info if it has been set
        if info:
            cls.trace_info[rank][type_] = info
        else:
            cls.trace_info[rank][type_] = []

    @classmethod
    def _writeTrace(cls, rank, type_):
        """Write the trace to the file

        Parameters
        ----------

        rank: str
            The rank whose trace to write

        type_: str
            Description of the information this trace is collecting

        Returns
        -------

        None

        """
        with open(cls.prefix + "-" + rank + "-" + type_ + ".csv", "a") as f:
           for use in cls.traces[rank][type_]:
                f.write(use)

        cls.traces[rank][type_] = []
