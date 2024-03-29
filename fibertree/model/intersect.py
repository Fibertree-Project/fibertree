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

        """
        assert len(traces) == 1

        new_intersects = len(traces[0])

        # Throw away the header, since we don't need it
        if not self.started:
            self.started = True
            new_intersects -= 1

        self.num_intersects += new_intersects

class SkipAheadIntersector(Intersector):
    """Class for counting intersections with a skip-ahead-intersector"""

    def addTraces(self, *traces):
        """Consume the information in the trace to count intersections

        Parameters
        ----------

        traces: list
            Lists of entries added since the last call to addTraces

        Returns
        -------

        None

        Note: Must be called after two fibers are fully intersected

        """
        assert len(traces) == 2

        trace0 = traces[0]
        trace1 = traces[1]

        # Throw away the header, since we don't need it
        if not self.started and trace0:
            self.started = True

            self.num_ranks = (len(trace0[0]) - 1) // 2

            trace0 = trace0[1:]
            trace1 = trace1[1:]

        i0 = -1
        i1 = -1

        def get_next(trace, i):
            if i + 1 < len(trace):
                return trace[i + 1][self.num_ranks:self.num_ranks * 2], i + 1
            return None, None

        point0, i0 = get_next(trace0, i0)
        point1, i1 = get_next(trace1, i1)

        if point0 is None or point1 is None:
            return

        assert point0 is not None and point1 is not None and point0[:-1] == point1[:-1]

        fiber = point0[:-1]
        curr = None

        while point0 and point1:
            if point0 == point1:
                self.num_intersects += 1
                curr = None

                point0, i0 = get_next(trace0, i0)
                point1, i1 = get_next(trace1, i1)

            elif point0 < point1:
                if curr != 0:
                    curr = 0
                    self.num_intersects += 1

                point0, i0 = get_next(trace0, i0)

                # If we have reached the end of this iteration, forward the other
                # finger to the next fiber
                if point0 is None or fiber != point0[:-1]:
                    point1, i1 = get_next(trace1, i1)

            # point0 > point1
            else:
                if curr != 1:
                    curr = 1
                    self.num_intersects += 1

                point1, i1 = get_next(trace1, i1)

                # If we have reached the end of this iteration, forward the other
                # finger to the next fiber
                if point1 is None or fiber != point1[:-1]:
                    point0, i0 = get_next(trace0, i0)

            old_fiber = fiber
            if point0:
                fiber = point0[:-1]
            else:
                fiber = None

            # Do not need to check point1 because both fingers should fall of
            # the end of the trace at the same time

            if fiber != old_fiber:
                curr = None

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

        """
        assert len(traces) == 2

        trace0 = traces[0]
        trace1 = traces[1]

        # Throw away the header, since we don't need it
        if not self.started:
            self.started = True

            self.num_ranks = (len(trace0[0]) - 1) // 2

            trace0 = trace0[1:]
            trace1 = trace1[1:]

        i0 = -1
        i1 = -1

        def get_next(trace, i):
            if i + 1 < len(trace):
                return trace[i + 1][self.num_ranks:self.num_ranks * 2], i + 1
            return None, None

        point0, i0 = get_next(trace0, i0)
        point1, i1 = get_next(trace1, i1)

        if point0 is None or point1 is None:
            return

        assert point0 is not None and point1 is not None and point0[:-1] == point1[:-1]

        fiber = point0[:-1]

        while point0 and point1:
            self.num_intersects += 1

            if point0 == point1:
                point0, i0 = get_next(trace0, i0)
                point1, i1 = get_next(trace1, i1)

            elif point0 < point1:
                point0, i0 = get_next(trace0, i0)

                # If we have reached the end of this iteration, forward the other
                # finger to the next fiber
                if point0 is None or fiber != point0[:-1]:
                    point1, i1 = get_next(trace1, i1)


            # point0 > point1
            else:
                point1, i1 = get_next(trace1, i1)

                # If we have reached the end of this iteration, forward the other
                # finger to the next fiber
                if point1 is None or fiber != point1[:-1]:
                    point0, i0 = get_next(trace0, i0)

            if point0:
                fiber = point0[:-1]
            else:
                fiber = None

