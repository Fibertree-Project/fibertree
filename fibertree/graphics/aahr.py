"""AAHR Module"""

import logging

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.core.aahr')


class AAHR():
    """The AAHR class is used to define an "axis aligned hyper-rectangle"
    based on the points in the open range between upper left and lower
    right corners of the hyper-rectangle. It is useful for defining a
    range of points for highlighting.

    TBD: Allow a highlight point to be an AAHR. A naive way to do that
    would be to expand the AAHR in canonicalizeHighlights()....

    """

    def __init__(self, upper_left, lower_right):

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.core.aahr')

        self.length = len(upper_left)

        assert self.length == len(lower_right)

        self.upper_left = upper_left
        self.lower_right = lower_right


    def expand(self):
        """Expand AAHR into all its points

        """

        region = [()]

        for start, end in zip(self.upper_left, self.lower_right):
            region = self._cross(region, range(start, end))

        return region

    @staticmethod
    def _cross(a, b):

        result = []

        for i in a:
            for j in b:
                result.append(i + (j,))

        return result
        

    def __contains__(self, point):

        if not isinstance(point, tuple) or len(point) != self.length:
            return False

        for p, u, l in zip(point, self.upper_left, self.lower_right):
            if p < u or p >= l:
                return False

        return True
