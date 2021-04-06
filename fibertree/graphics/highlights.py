"""Highlight Module"""

import logging

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.graphics.highlights')



class HighlightManager():
    """HighlightManager """

    def __init__(self, highlights={}, highlight_subtensor={}, parent=None, level=None):

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.graphics.highlights')

        self.highlights = highlights
        self.highlight_subtensor = highlight_subtensor

        self.parent = parent
        self.level = level

        self.current_coord = None
        self.highlight_coords = {}

        #
        # The points to highlight for each worker at this level are
        # based on the first coordinate in each point, unless the
        # coordinate is a wildcard ('?')
        #
        self.active_coords = {}
    
        for worker, points in highlights.items():

            active_coords_temp = []

            for point in points:
                if point[0] not in ['?']:
                    active_coords_temp.append(point[0])

            self.active_coords[worker] = set(active_coords_temp)


    def addFiber(self, c):
        #
        # For each payload that was a fiber we need to recurse, but we
        # also need to figure out what to highlight at the next level
        # So these variables hold the highlight information with one
        # less coordinate (in "highlights_next") for each worker, and
        # a dictionary of workers (in "highlight_subtensor_next") that
        # are highlighting the remaining levels of the subtensor
        #
        self.current_coord = c

        highlights = self.highlights
        highlight_subtensor = self.highlight_subtensor
        active_coords = self.active_coords

        highlights_next = {}
        highlight_subtensor_next = {}

        for worker, points in highlights.items():
            #
            # Once we start highlighting a fiber, highlight the entire subtensor.
            #
            # TBD: Maybe we should have just copied highlight_subtensor
            #
            if worker in highlight_subtensor:
                highlight_subtensor_next[worker] = True

            #
            # Create the tail of the highlight coordinates as the next
            # highlights
            #
            highlights_next[worker] = []

            for point in points:
                len_point = len(point)

                #
                # If there are more than one coordinate in the point,
                # then add the remaining coordinates to the next
                # highlights
                #
                if len_point > 1 and (point[0] == c or point[0] == '?'):
                    highlights_next[worker].append(point[1:])

                #
                # If this was the last coordinate
                # maybe start highlighting a subtensor
                #
                if len_point == 1 and point[0] == c and c in active_coords[worker]:
                    highlight_subtensor_next[worker] = True
                    self.addHighlight(worker)


        highlight_manager_next = HighlightManager(highlights_next,
                                                  highlight_subtensor_next,
                                                  self,
                                                  self.level-1)

        return highlight_manager_next

    def addHighlight(self, worker):

        if not worker in self.highlight_coords:
            self.highlight_coords[worker] = set([self.current_coord])
        else:
            self.highlight_coords[worker].add(self.current_coord)

        parent = self.parent
        if parent is not None:
            parent.addHighlight(worker)


    def getColorCoord(self, c):

        #
        # For level 0, the highlight coords are the active coords and
        # tell the parent which of this child's workers were
        # highlighted
        #
        if self.level <= 0:
            self.highlight_coords = self.active_coords

            for worker, coords in self.highlight_coords.items():
                if c in coords:
                    # print(f"highlights[{worker}] = {self.highlights[worker]}")
                    parent = self.parent
                    if parent is not None:
                        self.parent.addHighlight(worker)


        color_coord = set([worker for worker, coords in self.highlight_coords.items() if c in coords])

        return color_coord

    def getColorSubtensor(self):

        color_subtensor = set([worker for worker in self.highlight_subtensor.keys()])
        return color_subtensor


    @staticmethod
    def canonicalizeHighlights(highlights, worker="PE"):
        """canonicalizeHighlights

        In methods that accept highlights there is considerable
        flexibility in the form that the highlights are provided. This
        method converts any of those forms into the canonical form,
        using keyword "worker" to assign a worker if one isn't
        provided in the "highlights" argument.  The canonical form is
        a dictionary of workers and lists of their highlighted points:


        {worker0: [(point0_coord0, point0_coord1, ...),
                   (point1_coord0, point1_coord1, ...),
                    ...],
         worker1: [(point0_coord0, point0_coord1, ...),
                   (point1_coord0, point1_coord1, ...),
                   ...],
         ...,
        }


        Alternative forms:

        1) Single point per worker

        {worker0: (point0_coord0, point0_coord1, ...),
         worker1: (point0_coord0, point0_coord1, ...),
          ...
        }


        2) List of points, no worker

        [(point0_coord0, point0_coord1, ...),
         (point1_coord0, point1_coord1, ...),
         ...]


        3) Single point, no worker

        (point1_coord0, point1_coord1, ...)


        Warning: if a coordinate is a tuple there is ambiguity in forms
        1 and 3, so they cannot be used.


        Parameters:
        -----------

        highlights: dictionary, list or tuple
        A specification of highlights, maybe not in canonical form

        worker: string
        A name to use for the worker, if highlights doesn't include one

        Returns:
        --------

        highlights: dictionary
        A specification of highlights in canonical form


        Raises:
        -------

        Nothing


        Bugs:
        -----

        A single point with a character as a coordinate is
        misinterpreted as a list of points

        """

        if not isinstance(highlights, dict):
            #
            # Massage highlights into proper form
            #
            highlights = {worker: highlights}

        #
        # Wrap highlights specified as a single point into a list
        #
        for pe, pe_highlights in highlights.items():
            #
            # If highlights is a single point convert to list
            #
            if len(pe_highlights):
                try:
                    temp = pe_highlights[0][0]

                except Exception:
                    temp = pe_highlights
                    pe_highlights = []
                    pe_highlights.append(temp)
                    highlights[pe] = pe_highlights

        return highlights

