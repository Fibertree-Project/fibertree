#cython: language_level=3
# cython: profile=True
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
    rank_matches = {}
    traces = {}

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def addUse(cls, rank, coord, pos, type_="iter", iteration_num=None):
        """Add a use of all tensors at the given rank and coord

        Parameters
        ----------

        rank: str
            The name of the rank

        coord: int
            The coordinate at this rank

        type_: str
            Description of the information this trace is collecting

        iteration_num: Optional[Tuple[int...]]
            The iteration number this use occurs on
            If none, use the default iteration number

        Returns
        -------

        None

        """
        assert cls.collecting
        assert rank in cls.line_order or rank in cls.rank_matches

        # Update the point
        if rank in cls.line_order:
            i = cls.line_order[rank]
            cls.point[i] = coord

        # Otherwise, set i based on the rank this rank matches
        else:
            i = cls.line_order[cls.rank_matches[rank]]

        # Make sure we are tracking this rank and type_
        if rank not in cls.traces.keys():
            return

        if type_ not in cls.traces[rank].keys():
            return

        if iteration_num is None:
            iteration_num = cls.iteration


        iteration = iteration_num[:(i + 1)]
        point = cls.point[:i] + [coord]

        data = iteration + point + [pos]
        data_str = ",".join(map(str, data))
        cls.traces[rank][type_][0].append(data_str + "\n")

        # If we are at the limit of the number of cached uses, write the data
        # to disk
        if len(cls.traces[rank][type_][0]) == cls.num_cached_uses:
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
        cls.rank_matches = {}
        cls.traces = {}

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
        return cls.iteration

    @classmethod
    def getLabel(cls, rank):
        """Get a new label for a fiber at this rank

        Parameters
        ----------

        rank: str
            The rank whose fiber we want to label

        loop_rank: str
            The rank we want to

        Returns
        -------

        None

        """
        assert cls.collecting

        if rank in cls.line_order:
            iter_rank = rank

        elif rank in cls.rank_matches:
            iter_rank = cls.rank_matches[rank]

        # If the correct rank has not been registered yet, create a new
        # Fiber label and combine later
        elif rank in cls.fiber_label:
            iter_rank = rank

        else:
            cls.fiber_label[rank] = 0
            iter_rank = rank

        cls.fiber_label[iter_rank] += 1
        return cls.fiber_label[iter_rank] - 1

    @classmethod
    def getIndex(cls, rank):
        """Get the index in the line order of this rank

        Parameters
        ----------

        rank: str
            The rank whose index to get

        Returns
        -------

        None

        """
        assert cls.collecting
        assert rank in cls.line_order or rank in cls.rank_matches

        if rank in cls.line_order:
            return cls.line_order[rank]
        else:
            return cls.line_order[cls.rank_matches[rank]]

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
        assert line in cls.line_order or line in cls.rank_matches

        if line in cls.line_order.keys():
            cls.iteration[cls.line_order[line]] += 1

        else:
            cls.iteration[cls.line_order[cls.rank_matches[line]]] += 1

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
    def isTraced(cls, rank, type_):
        """Returns True if the given rank and trace is actually being collected

        Parameters
        ----------

        rank: str
            The rank name in question

        type_: str
            The name of the trace in question

        """
        assert cls.collecting

        return rank in cls.traces and type_ in cls.traces[rank]

    @classmethod
    def matchRanks(cls, rank1, rank2):
        """Register the fact that rank1 and rank2 are associated with the same
        level of the loop nest

        Parameters
        ----------

        rank1, rank2: str
            Ranks to match

        Returns
        -------

        None

        """
        assert rank1 in cls.line_order or rank2 in cls.line_order

        if rank1 in cls.line_order:
            loop_rank = rank1
            new_rank = rank2

        else:
            loop_rank = rank2
            new_rank = rank1

        cls.rank_matches[new_rank] = loop_rank

        # Combine fiber labels
        if new_rank in cls.fiber_label:
            cls.fiber_label[loop_rank] += cls.fiber_label[new_rank]
            del cls.fiber_label[new_rank]

        # Start any traces that can now be started
        if new_rank in cls.traces:
            for type_ in cls.traces[new_rank]:
                if not cls.traces[new_rank][type_][1]:
                    cls._startTrace(new_rank, type_)

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
        assert rank in cls.line_order or rank in cls.rank_matches

        if rank in cls.line_order:
            end = cls.line_order[rank] + 1
        else:
            end = cls.line_order[cls.rank_matches[rank]] + 1

        cls.traces[rank][type_] = ([], True)

        with open(cls.prefix + "-" + rank + "-" + type_ + ".csv", "w") as f:
            f.write("")

        pos = ",".join(r + "_pos" for r in cls.loop_order[:end])
        coord = ",".join(r for r in cls.loop_order[:end])

        headings = [pos, coord, "fiber_pos"]
        cls.traces[rank][type_][0].append(",".join(headings) + "\n")


    @classmethod
    def trace(cls, rank, type_="iter"):
        """Set a rank to trace

        Note must be called after Metrics.beginCollect()

        Parameters
        ----------

        rank: str
            The rank to collect the trace of

        type_: str
            Description of the information this trace is collecting

        Returns
        -------

        None

        """
        assert cls.prefix is not None
        assert cls.collecting

        if rank not in cls.traces.keys():
            cls.traces[rank] = {}

        cls.traces[rank][type_] = ([], False)

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
            f.write("".join(cls.traces[rank][type_][0]))

        cls.traces[rank][type_] = ([], True)
