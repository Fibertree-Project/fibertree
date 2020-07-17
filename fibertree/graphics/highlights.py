class HighlightManager():
    """HighlightManager """

    def __init__(self, highlights={}, highlight_subtensor={}):

        self.highlights = highlights
        self.highlight_subtensor = highlight_subtensor

        #
        # The points to highlight for each worker at this level are
        # based on the first coordinate in each point
        #
        self.highlight_coords = {}
    
        for worker, points in highlights.items():
            self.highlight_coords[worker] = [pt[0] for pt in points]


    def addFiber(self, c):
        #
        # The each payload that was a fiber we need to recurse, but we
        # also need to figure out what to highlight at the next level
        # So these variables hold the highlight information with one
        # less coordinate (in "highlight_next") for each worker, and a
        # dictionary of workers (in "highlight_subtensor_next") that
        # are highlighting the remaining subtensor
        #
        highlights = self.highlights
        highlight_subtensor = self.highlight_subtensor
        highlight_coords = self.highlight_coords

        highlight_next = {}
        highlight_subtensor_next = {}

        for worker, points in highlights.items():
            #
            # Create the tail of the highlight coordinates
            #
            highlight_next[worker] = [pt[1:] for pt in points if len(pt) > 1 and pt[0] == c]
            #
            # Once we start highlighting a fiber, highlight the entire subtensor.
            # TBD: Maybe we should have just copied highlight_subtensor
            #
            if worker in highlight_subtensor:
                highlight_subtensor_next[worker] = True

            #
            # If there are no more coordinates,
            # maybe start highlighting a subtensor
            #
            if len(highlight_next[worker]) == 0 and c in highlight_coords[worker]:
                highlight_subtensor_next[worker] = True

        highlight_manager_next = HighlightManager(highlight_next, highlight_subtensor_next)

        return highlight_manager_next

    def getColorCoord(self, c):

        color_coord = set([worker for worker, coords in self.highlight_coords.items() if c in coords])
        return color_coord

    def getColorSubtensor(self):

        color_subtensor = set([worker for worker in self.highlight_subtensor.keys()])
        return color_subtensor

