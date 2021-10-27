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
    metrics = []

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def beginCollect(cls):
        """Begin metrics collection

        Start collecting metrics during future HFA program execution.

        Parameters
        ----------

        None

        Returns
        -------

        None

        """
        cls.metrics.append({})
        cls.collecting = True

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
        return cls.metrics[-1]

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
        cls.collecting = False

    @classmethod
    def inc(cls, line, metric, inc):
        """Increment a metric during collection

        Increment the given metric associated with the given line of code by
        the specified amount. If the line or metric is not already being
        tracked, they are first added to the dictionary of metrics.

        If metrics collection is off, this function returns without modifying
        anything.

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

        if not cls.collecting:
            return

        line = line.strip()

        if line not in cls.metrics[-1]:
            cls.metrics[-1][line] = {}

        if metric not in cls.metrics[-1][line]:
            cls.metrics[-1][line][metric] = 0

        cls.metrics[-1][line][metric] += inc


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

