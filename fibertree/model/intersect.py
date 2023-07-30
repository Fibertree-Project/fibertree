#cython: language_level=3
"""
Compute the number of intersection attempts for different intersection styles
"""

class Intersector:
    """Superclass for counting intersections"""

    def __init__(self):
        """Create an Intersector

        Parameters
        ----------

        None

        """
        self.num_intersects = 0

    def addTrace(self, trace):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        trace: list
            List of entries added since the last call to addTrace

        Returns
        -------

        None

        Note: Should be implemented by each subclass

        """
        raise NotImplementedError

    def getNumIntersects(self):
        """Get the number of intersection tests performed so far

        Parameters
        ----------

        None

        Returns
        -------

        None

        """
        return self.num_intersects

class LeaderFollowerIntersector(Intersector):
    """Class for counting intersections with a leader-follower intersector"""

    def __init__(self):
        """Create a LeaderFollowerIntersector

        Parameters
        ----------

        None

        """
        super().__init__()
        self.started = False

    def addTrace(self, trace):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        trace: list
            List of entries added since the last call to addTrace

        Returns
        -------

        None

        Note: Should be implemented by each subclass

        """
        new_intersects = len(trace)
        if not self.started:
            self.started = True
            new_intersects -= 1

        self.num_intersects += new_intersects

