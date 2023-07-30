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
        self.started = False

    def addTraces(self, *traces):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        traces: list
            Lists of entries added since the last call to addTraces

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

    def addTraces(self, *traces):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        traces: list
            Lists of entries added since the last call to addTraces

        Returns
        -------

        None

        Note: Should be implemented by each subclass

        """
        assert len(traces) == 1

        new_intersects = len(traces[0])

        # Throw away the header, since we don't need it
        if not self.started:
            self.started = True
            new_intersects -= 1

        self.num_intersects += new_intersects

class TwoFingerIntersector(Intersector):
    """Class for counting intersections with a two-finger-intersector"""

    def addTraces(self, *traces):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        traces: list
            Lists of entries added since the last call to addTraces

        Returns
        -------

        None

        Note: Should be implemented by each subclass

        """
        assert len(traces) == 2

        trace0 = traces[0]
        trace1 = traces[1]

        # Throw away the header, since we don't need it
        if not self.started:
            self.started = True

            trace0 = trace0[1:]
            trace1 = trace1[1:]

            self.num_ranks = (len(trace0[0]) - 1) // 2

        i0 = -1
        i1 = -1

        def get_next(trace, i):
            if i + 1 < len(trace):
                return trace[i + 1][self.num_ranks:self.num_ranks * 2], i + 1
            return None, None

        point0, i0 = get_next(trace0, i0)
        point1, i1 = get_next(trace1, i1)

        while point0 and point1:
            self.num_intersects += 1

            if point0 == point1:
                point0, i0 = get_next(trace0, i0)
                point1, i1 = get_next(trace1, i1)

            elif point0 < point1:
                point0, i0 = get_next(trace0, i0)

            # point0 > point1
            else:
                point1, i1 = get_next(trace1, i1)

